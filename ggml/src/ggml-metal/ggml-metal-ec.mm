// ggml-metal-ec.mm — Metal implementation of the expert-slot-cache hook table (ggml-ec-hooks.h).
//
// Port of ggml-cuda/expert-cache.cu to Apple Metal, per the approved primitive table:
//   cudaHostAlloc staging        -> 2 rotating StorageModeShared MTLBuffers + blit/replaceRegion
//                                   (host-writable so pread can fill them; CUDA ring was 4-deep,
//                                   the design table specifies double-buffered = 2)
//   cudaMalloc(C*slab) arena     -> newBufferWithLength(C*slab, MTLResourceStorageModePrivate)
//   side-stream cudaMemcpyAsync  -> dedicated id<MTLCommandQueue> (copyQ) + per-slot MTLSharedEvent
//   slot-ids ring (pinned 4-deep)-> 4 private destination MTLBuffers fed from host-mapped shared
//                                   source arrays (contents mapped once, host-filled per decode call);
//                                   the blit on the device queue preserves CUDA's stream-ordering
//                                   contract (deviation note in ec-metal-port-REPORT.md)
//   LRU+stamp eviction 3-pass    -> ported verbatim from expert-cache.cu (pure CPU logic)
//   disk source                  -> pread(fd, staging contents, slab, offs + eid*slab), reading the
//                                   SAME ggml_backend_ec_get_file_source registry as CUDA
//
// Ordering model: command buffers committed on one MTLCommandQueue execute in commit order —
// Metal's equivalent of CUDA stream ordering. "fork" = signal event via a command buffer committed
// at issue time; "join" = commit a wait-only command buffer before the graph encodes this call.
// cudaEventQuery has no Metal equivalent: completion-handler pending counters mirror its semantics
// (advisory only, better eviction victims — same as CUDA). Blocking waits use dispatch_semaphore
// only on the rare all-pending path. Compiled manual-reference-counted like the rest of ggml-metal
// (the tree does not enable -fobjc-arc anywhere).
//
// Default OFF: ggml_backend_ec_hooks_ptr stays NULL unless LLAMA_EXPERT_CACHE > 0 is seen at the
// lazy device-registry probe (env can land after device enumeration during model load — the same
// lesson CUDA learned; ec_n_cached() re-probes until positive).

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <dispatch/dispatch.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "../ggml-ec-hooks.h"
#include "ggml-metal-device.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>    // disk source: open
#include <unistd.h>   // disk source: pread
#include <unordered_map>
#include <vector>

static int ec_n_cached(void);   // fwd
// [gate59-64/73/75] THE single serve-eligibility predicate: a mul_mat_id node may live on the GPU
// (bounded arena copy + served intercept) only when the graph is decode-shaped (ids ne[1]==1,
// width<=C) AND its weights are registry-known. Anything else — prefill, chunks, MTP verify at
// C<elems — must run CPU-side: on GPU it would get stock full-size copies of all 46 expert
// tensors (~16.5 GB of MTL0 allocs + upload; the repeated wired blowups). CPU reads the GGUF
// mmap directly, zero copies, private stays bounded. consulted by: offload_op (ggml-metal.cpp),
// assign_gpu, moe_allow_orig_src, force_copy (via the sched graph scan).
static int ec_node_serve_eligible(const ggml_tensor * node) {
    if (!node || node->op != GGML_OP_MUL_MAT_ID || !node->src[0] || !node->src[0]->name[0]) return 0;
    const ggml_tensor * ids = node->src[2];
    if (!ids || ids->ne[1] != 1 || ids->ne[0] < 1 || ids->ne[0] > ec_n_cached()) return 0;
    ggml_ec_file_src fs;
    return ggml_backend_ec_get_file_source(node->src[0]->name, &fs) ? 1 : 0;
}
int ggml_metal_ec_node_serve_eligible(const ggml_tensor * node) { return ec_node_serve_eligible(node); }
static int hook_assign_gpu(const ggml_tensor * node) {
    // entries only exist after the first copy phase; decide from the loader registry instead
    return ec_node_serve_eligible(node);
}
static int g_ec_n = -1; // C slots per entry (layer); parsed LAZILY from LLAMA_EXPERT_CACHE on each
                        // query — model-params setenv() can land after device enumeration, so a
                        // one-shot read at init would miss it (CUDA lesson, kept verbatim).
static int ec_n_cached(void) {
    if (g_ec_n <= 0) {
        const char * env = getenv("LLAMA_EXPERT_CACHE");
        g_ec_n = env ? atoi(env) : 0;
    }
    return g_ec_n;
}

static std::atomic<uint64_t> g_ec_hits{0}, g_ec_misses{0}, g_ec_evicts{0}, g_ec_calls{0};
static std::atomic<uint64_t> g_ec_fallbacks{0};
static std::atomic<uint64_t> g_ec_intercepts{0}, g_ec_declines{0}, g_ec_setups{0}, g_ec_fillids{0};
static std::atomic<uint64_t> g_ec_copy_bytes{0}, g_ec_side_copies{0};
static std::atomic<uint64_t> g_ec_verify_ok{0}, g_ec_verify_bad{0}, g_ec_hoststale{0};
static std::atomic<uint64_t> g_ec_audit_ok{0}, g_ec_audit_bad{0};

#define EC_ID_RING 4
// pending-counter helpers: plain int + __atomic builtins keep structs movable/copyable (std::atomic
// members would delete the move ctor that unordered_map rehashing needs). Semantics identical to
// cudaEventQuery(ev) == cudaErrorNotReady — advisory only, same as CUDA's in_flight events.
#define PEND_LOAD(p)   __atomic_load_n((p), __ATOMIC_RELAXED)
#define PEND_ADD(p, n) __atomic_add_fetch((p), (n), __ATOMIC_RELAXED)

#define EC_STAGE_RING 8        // [perf] 8-deep staging: lets fill_ids keep ~7 preads in flight while
                               // earlier slabs blit; 2-deep serialized NVMe under blit latency (gen 4.8)
// --- [policy] global routing-frequency table + optional profile file (CUDA-twin semantics) ------
// freq[L*128+E] counts routes observed at env-gated; seeds hot-pins at setup and (LLAMA_EC_PROFILE)
// persists across runs. Routing-transparent: changes WHICH experts stay resident, never WHICH route.
#define EC_MAX_LAYERS 64
#define EC_N_EXPERT_MAX 128
static std::atomic<uint32_t> g_ec_freq[EC_MAX_LAYERS * EC_N_EXPERT_MAX];
static int g_ec_pin_k     = 0;    // LLAMA_EC_PIN=K enables hot-pinning

static void ec_profile_load(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) return;
    for (int l = 0; l < EC_MAX_LAYERS; l++) {
        for (int e = 0; e < EC_N_EXPERT_MAX; e++) {
            uint32_t v = 0;
            if (fread(&v, 4, 1, f) != 1) { fclose(f); return; }
            g_ec_freq[l*EC_N_EXPERT_MAX + e].store(v, std::memory_order_relaxed);
        }
    }
    fclose(f);
    fprintf(stderr, "[metal-ec] profile loaded: %s\n", path);
}
static void ec_profile_save(void) {
    const char * path = getenv("LLAMA_EC_PROFILE");
    if (!path || !path[0]) return;
    char tmp[1024]; snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE * f = fopen(tmp, "wb");
    if (!f) return;
    for (int l = 0; l < EC_MAX_LAYERS; l++)
        for (int e = 0; e < EC_N_EXPERT_MAX; e++) {
            uint32_t v = g_ec_freq[l*EC_N_EXPERT_MAX + e].load(std::memory_order_relaxed);
            fwrite(&v, 4, 1, f);
        }
    fclose(f);
    rename(tmp, path);
    fprintf(stderr, "[metal-ec] profile saved: %s\n", path);
}

// --- global staging ring (one for every entry; sized to the largest slab seen) --------------
static id<MTLBuffer>           g_stage[EC_STAGE_RING];
static int                     g_stage_pend[EC_STAGE_RING];
static dispatch_semaphore_t    g_stage_done[EC_STAGE_RING];
static int                     g_stage_next = 0;
static uint64_t                g_stage_len  = 0;
static void ec_stage_ring_ensure(id<MTLDevice> device, uint64_t slab) {
    if (slab <= g_stage_len) return;                    // already sized for this slab
    static id<MTLDevice> s_dev = nil;
    if (!s_dev) { s_dev = device; [s_dev retain]; }
    // drain: never drop buffers an in-flight blit may still be reading (resize happens at most a
    // couple of times — slabs take a handful of distinct sizes across layers)
    for (int i = 0; i < EC_STAGE_RING; i++) {
        while (PEND_LOAD(&g_stage_pend[i]) > 0 && g_stage_done[i]) {
            dispatch_semaphore_wait(g_stage_done[i], DISPATCH_TIME_FOREVER);
        }
    }
    for (int i = 0; i < EC_STAGE_RING; i++) {
        if (g_stage[i]) { [g_stage[i] release]; g_stage[i] = nil; }
        g_stage[i] = [s_dev newBufferWithLength:(size_t) slab options:MTLResourceStorageModeShared];
        [g_stage[i] retain];
        g_stage_pend[i] = 0;                             // drained: only call between synced phases
        if (!g_stage_done[i]) g_stage_done[i] = dispatch_semaphore_create(0);
    }
    g_stage_len = slab;
}
#define EC_RING_CAP_INTS 512   // fixed ring capacity (decode-sized); grow path deferred to v1.1

// pending-counter helpers: plain int + __atomic builtins keep structs movable/copyable (std::atomic
// members would delete the move ctor that unordered_map rehashing needs). Semantics identical to
// cudaEventQuery(ev) == cudaErrorNotReady — advisory only, same as CUDA's in_flight events.

// [perf] LLAMA_EC_PERF=1: aggregate time-in-hook stats, dumped at exit.
#include <chrono>
struct ec_perf {
    std::atomic<uint64_t> pread_ns{0}, pread_n{0};
    std::atomic<uint64_t> gate_ns{0}, gate_n{0};
    std::atomic<uint64_t> blit_ns{0}, blit_n{0};          // submit side (fork+encode+commit)
    std::atomic<uint64_t> join_ns{0}, join_n{0};          // per-slot join wait in fill_ids
    std::atomic<uint64_t> fillids_ns{0}, fillids_n{0};    // whole fill_ids hook
    std::atomic<uint64_t> hit_ns{0}, hit_n{0};            // resident-slot path (no fill)
    std::atomic<uint64_t> intercept_ns{0}, intercept_n{0};
    bool on() const { return getenv("LLAMA_EC_PERF") != nullptr; }
};
static ec_perf g_perf;
static inline uint64_t ec_now_ns() {
    using namespace std::chrono;
    return (uint64_t) duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}
struct ec_t { std::atomic<uint64_t> * dst; std::atomic<uint64_t> * n; uint64_t t0;
             ec_t(std::atomic<uint64_t> & d, std::atomic<uint64_t> & c) : dst(&d), n(&c) { t0 = ec_now_ns(); }
             ~ec_t() { dst->fetch_add(ec_now_ns() - t0, std::memory_order_relaxed); n->fetch_add(1, std::memory_order_relaxed); } };


// MTLSharedEvent wrapper with a cudaEventQuery-mirror pending counter. Value semantics mirror
// ggml_metal_event in ggml-metal-device.m (monotonic signal values).
struct ec_evt {
    id<MTLSharedEvent>     ev;      // retained for entry lifetime
    uint64_t               value;   // accessed via __atomic builtins
    int                    pending; // signals committed but not yet executed (in flight)
    dispatch_semaphore_t   done;    // posted by completion handler; rare bounded waits only
};

// --- [async] speculative background fill ---------------------------------------------------
// Demand reads STAY on the decode thread inside the scheduler copy phase (before the graph
// commit) — correctness first. A background worker instead spends the GPU-compute window
// re-residing recently-evicted experts (skewed routing => reuse distance ~W > C), converting
// tomorrow's demand misses into hits. Metadata (slot state, slot_of_id, LRU ticks, job deque)
// lives under one global mutex; the pread itself runs UNLOCKED against a slot reserved LOADING
// by its owner — worker and demand can therefore never duplicate a read nor expose half-written
// bytes. Routing-transparent by construction: only slot PLACEMENT changes, ids never — greedy
// output must stay byte-identical with the worker on or off.
struct ggml_metal_ec_entry;
enum ec_slot_st : uint8_t { EC_ST_FREE = 0, EC_ST_LOADING = 1, EC_ST_READY = 2 };
enum ec_job_src : uint8_t { EC_JOB_PF = 0, EC_JOB_SWEEP = 1 };   // both advisory: drop if stale
struct ec_job { ggml_metal_ec_entry * e; int32_t eid; };
static std::mutex              g_ec_mtx;
static std::condition_variable g_ec_cv;
static std::deque<ec_job>      g_ec_jobs;          // bounded, advisory
static std::atomic<bool>       g_ec_quit{false};
// [demand-parallel] the same pool also serves DEMAND fills: the served-copy phase posts every
// need_copy slot of a layer as a high-priority job (demand=1: the slot is ALREADY claimed by
// demand — the worker only reads bytes, demand settles), then barrier-waits for the layer.
// N workers x ~5ms/pread compresses the ~110-serial-pread/token chain toward 110/N.
struct ec_djob { ggml_metal_ec_entry * e; int32_t eid; int64_t slot; };
static std::deque<ec_djob>     g_ec_djobs;         // high priority: drained before PF jobs
static std::atomic<int>        g_ec_dpend{0};      // posted-not-yet-settled demand jobs
static std::condition_variable g_ec_dcv;
static int                     g_nworkers = 4;
static std::atomic<uint64_t>   g_dbar_ns{0}, g_dbar_n{0}, g_dwave_n{0};  // [diag] barrier wall
static bool                    g_async_on = false; // set in ggml_backend_ec_hooks_register_metal
static std::atomic<uint64_t>   g_ec_pf_hit{0}, g_ec_pf_fill{0}, g_ec_pf_drop{0};
static std::mutex                          g_ec_entries_mtx;
static std::vector<ggml_metal_ec_entry *>  g_ec_entries;   // stable addresses (map values)
static bool ec_pf_claim_locked(ggml_metal_ec_entry * e, int32_t eid, int64_t * out);
static void ec_pf_enqueue(ggml_metal_ec_entry * e, int32_t eid);
static void * ec_worker_thread(void *);
static void ec_start_worker_once(void);

static ec_evt * evt_init(id<MTLDevice> dev) {
    ec_evt * e = new ec_evt();
    e->ev = [dev newSharedEvent];
    [e->ev retain];
    e->value  = 0;
    e->pending = 0;
    e->done   = dispatch_semaphore_create(0);
    return e;
}

static bool evt_in_flight(const ec_evt * e) { return PEND_LOAD(&e->pending) > 0; }

static void evt_signal(ec_evt * ev, id<MTLCommandBuffer> cb) {
    uint64_t v = __atomic_add_fetch(&ev->value, 1, __ATOMIC_RELAXED);
    [cb encodeSignalEvent:ev->ev value:v];
    PEND_ADD(&ev->pending, 1);
    dispatch_semaphore_t done = ev->done;
    // setCompletionHandler: is removed in recent macOS SDKs — use the completionHandler property
    [cb addCompletedHandler:^(id<MTLCommandBuffer> b){ GGML_UNUSED(b); PEND_ADD(&ev->pending, -1); dispatch_semaphore_signal(done); }];
}

static void evt_wait(ec_evt * ev, id<MTLCommandBuffer> cb) {
    [cb encodeWaitForEvent:ev->ev value:__atomic_load_n(&ev->value, __ATOMIC_RELAXED)];
}

struct ggml_metal_ec_slot {
    int32_t  expert_id = -1;      // expert currently resident (-1 = never used)
    ec_slot_st state = EC_ST_FREE;// [async] FREE | LOADING (owned: loading_eid set) | READY
    int32_t  loading_eid = -1;    // eid under pread while LOADING (owner: demand or worker)
    bool     was_prefetched = false; // worker-filled; counted the first time demand uses it
    uint64_t last_use  = 0;       // LRU tick (call generation)
    ec_evt * in_flight = nullptr; // slot-copy completion event (side queue)
    bool     loading   = false;   // legacy join flag for the blit fallback (cb_commit_wait)
    bool     hot       = false;   // pin segment (policy layer); worker may never evict these
    bool     bound_unfinished = false; // [async] bound into a committed graph whose logits sync
                                       // has not returned: worker must not recycle. Set when
                                       // fill_ids binds the ring; cleared at the NEXT fill_ids
                                       // for this entry — decode reaches fill_ids only after
                                       // sampling the previous token, a full device-sync proof.
    uint64_t stamp_gen = 0;       // generation of the call that last touched this slot
};

struct ggml_metal_ec_entry {
    const ggml_tensor * host_ptr  = nullptr;  // registry key: host expert tensor (stable)
    const ggml_tensor * dev_ptr   = nullptr;  // this call's device-side copy (input_cpy), may rotate
    const void        * host_base = nullptr;  // GGUF mmap base (cold fallback source)
    ggml_metal_device_t mdev = nullptr;       // Metal device owning buffers/queues

    // [disk source] loader-registered GGUF range (same registry as CUDA)
    bool     disk = false;
    int      fd   = -1;
    uint64_t file_offs = 0;

    // [perf] staging ring is global (mirrors CUDA's single pinned ring), created lazily in
    // ec_stage_ring_ensure(); per-entry rings at depth 2 serialized each layer's preads behind the
    // previous blit's completion handler — the gen-4.8-t/s bottleneck. Fields kept but unused.
    id<MTLBuffer> stage[1];                     // legacy: not allocated anymore
    int           stage_pending[1];
    dispatch_semaphore_t stage_done[1];
    int stage_next = 0;

    size_t   slab      = 0;   // nb[2] bytes per expert (identical math to CUDA)
    int64_t  n_expert  = 0;
    int64_t  C         = 0;
    id<MTLBuffer> arena = nullptr;             // private, C * slab, or the scheduler's bounded copy buffer
    size_t   arena_off = 0;                    // tensor offset inside the buffer when bounded
    bool     bounded   = false;                // ne[2] == C: this buffer IS the arena
    bool     own_arena = false;                // entry owns arena (no adoption): copy_slabs must route slots
    std::vector<ggml_metal_ec_slot> slots;
    std::vector<int64_t> slot_of_id;           // expert_id -> slot, -1 if not resident

    int      engaged = 0;
    int      partial_fill = 0;
    int      layer_idx    = -1;   // [policy] blk.N parsed at creation; -1 = freq/pin disabled
    int      prewarmed    = 0;    // [policy] top-k pins loaded once at first setup
    int64_t  last_repin   = -1;   // [policy] gen bucket the pins were last recomputed at
    // [async] worker handshake: teardown (stale-entry rail on model reload) must not yank
    // arena/fd from under an in-flight speculative pread. Teardown flips alive=0 under g_ec_mtx
    // (blocks new claims), drops queued jobs, then waits for in_pf to drain (bounded: one pread).
    // Plain ints + __atomic builtins: std::atomic members would delete the move ctor.
    int alive = 1;
    int in_pf = 0;
    int      dbg_trace = 0;   // per-entry trace budget (LLAMA_EC_DEBUG_SLOT_VERIFY)
    int      dbg_ids   = 0;
    int      dbg_audit = 0;
    int64_t  ids_expect = 0;
    uint64_t gen = 0;

    // slot-ids ring (design table row 4): 4 private destinations + host-mapped shared sources,
    // contents mapped once and host-filled per decode call. Blit on the device queue preserves
    // CUDA's stream-ordered upload contract. Ring depth 4, rotation per call.
    id<MTLBuffer> ring_dst[EC_ID_RING];        // private (kernels read this)
    id<MTLBuffer> ring_src[EC_ID_RING];        // shared, mapped over host arrays
    int32_t *     ring_host[EC_ID_RING];       // == ring_src contents
    int           ring_pending[EC_ID_RING];
    dispatch_semaphore_t ring_done[EC_ID_RING];
    int      ring_next = 0;
    int      last_ring_idx = -1;               // this call's ring entry (intercept binds this)
    int64_t  last_ids_n = 0;

    id<MTLCommandQueue> copyQ = nullptr;       // dedicated side command queue per entry
    ec_evt * fork_ev = nullptr;                // fork: copyQ <- device-queue ordering at issue time
};

static std::unordered_map<const ggml_tensor *, ggml_metal_ec_entry> g_entries;
static std::unordered_map<const ggml_tensor *, ggml_metal_ec_entry *> g_dev_idx;

static ggml_backend_ec_hooks g_hooks;

int ggml_metal_ec_active(void) {
    return ec_n_cached() > 0;
}

// --- cross-TU accessors for the ggml-metal.cpp MUL_MAT_ID intercept -------------------------

// Activations on a plain-CPU buffer have no Metal view, so kernels reading them get nil. During EC
// runs create the view lazily, once per buffer (no-copy when the base is page-aligned, else copy).
// Cost is activation-sized; EC weight tensors never reach here because the served path binds arena refs.
static id<MTLDevice> g_ec_device = nil;

void ggml_metal_ec_capture_device(ggml_metal_device_t dev) {
    if (g_ec_device) return;
    g_ec_device = (id<MTLDevice>) ggml_metal_device_get_obj(dev);
    [g_ec_device retain];
}

void * ggml_metal_ec_host_view(const struct ggml_tensor * t) {
    if (!ggml_backend_ec_hooks_ptr || !ggml_backend_ec_hooks_ptr->active() || !t->buffer || !g_ec_device) return NULL;

    static std::unordered_map<ggml_backend_buffer_t, id<MTLBuffer>> views;
    auto it = views.find(t->buffer);
    if (it != views.end()) return (void *) it->second;

    void * base = ggml_backend_buffer_get_base(t->buffer);
    size_t size = ggml_backend_buffer_get_size(t->buffer);
    if (!base || size == 0) return NULL;

    // hard rail: a host buffer this large is the whole-file GGUF mapping (model mmap), never an
    // activation. Viewing it burned 5 GB of wired per attempt and segfaulted (gates 69/70). EC weight
    // tensors must never reach here at all — the served path binds arena refs; if one does, declining
    // (NULL -> ggml_metal_buffer_get_id assert) is the loud, safe failure.
    if (size > (size_t) 2ull * 1024 * 1024 * 1024) {
        static bool refused = false;
        if (!refused) {
            refused = true;
            fprintf(stderr, "[metal-ec] host view refused: buffer %zu bytes (tensor %s)\n",
                    size, ggml_get_name(t) ? ggml_get_name(t) : "?");
        }
        return NULL;
    }

    id<MTLBuffer> view = [g_ec_device newBufferWithBytesNoCopy:base length:size options:MTLResourceStorageModeShared deallocator:nil];
    if (!view) {
        view = [g_ec_device newBufferWithBytes:base length:size options:MTLResourceStorageModeShared];
        if (!view) {
            fprintf(stderr, "[metal-ec] host view failed for buffer %p (%zu bytes)\n", base, size);
            return NULL;
        }
    }

    views[t->buffer] = [view retain];
    return (void *) views[t->buffer];
}

// [blocker A] sched split_graph never rewrites node->src[j]: at encode time the intercept sees the
// ORIGINAL host tensor, not the scheduler copy that g_dev_idx is keyed by. g_entries is keyed by that
// host pointer (lookup_or_create line: g_entries.emplace(host, ...)), so fall back to it on a dev_idx
// miss. Same entry either way — g_entries stores the entries, g_dev_idx stores pointers into it.
static ggml_metal_ec_entry * ec_entry_lookup(const ggml_tensor * t) {
    auto it = g_dev_idx.find(t);
    if (it != g_dev_idx.end()) return it->second;
    auto jt = g_entries.find(t);
    return jt != g_entries.end() ? &jt->second : nullptr;
}

struct ggml_metal_ec_ref ggml_metal_ec_arena_ref(const ggml_tensor * dev) {
    struct ggml_metal_ec_ref r = { nullptr, 0 };
    const ggml_metal_ec_entry * e = ec_entry_lookup(dev);
    if (e && e->arena) {
        r.metal = (void *) e->arena;
        r.offs  = e->arena_off;
    }
    return r;
}

struct ggml_metal_ec_ref ggml_metal_ec_ids_ref(const ggml_tensor * dev) {
    struct ggml_metal_ec_ref r = { nullptr, 0 };
    const ggml_metal_ec_entry * e = ec_entry_lookup(dev);
    if (e && e->last_ring_idx >= 0 && e->ring_dst[e->last_ring_idx]) {
        r.metal = (void *) e->ring_dst[e->last_ring_idx];
    }
    return r;
}

// graph-compatibility refusal mirror (CUDA: EC-served decode must be eager). The ops-file dispatch
// declines capture-time serves; this is the same gate exposed for schedulers/diagnostics.
int ggml_metal_ec_graph_refused(struct ggml_cgraph * gf) {
    if (!ggml_backend_ec_hooks_ptr || !ggml_backend_ec_hooks_ptr->active()) return 0;
    for (int i = 0; i < gf->n_nodes; i++) {
        const ggml_tensor * op = gf->nodes[i];
        if (op->op == GGML_OP_MUL_MAT_ID && op->src[0] &&
            ggml_backend_ec_hooks_ptr->engaged_this_call(op->src[0])) {
            return 1;
        }
    }
    return 0;
}

// --- command-buffer helpers ------------------------------------------------------------------

static id<MTLCommandQueue> device_queue(ggml_metal_ec_entry * e) {
    return (id<MTLCommandQueue>) ggml_metal_device_get_queue(e->mdev);
}

static void cb_commit_signal(ec_evt * ev, id<MTLCommandQueue> q) {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    evt_signal(ev, cb);
    [cb commit];
}

// join: order the device queue behind an event BEFORE the graph's command buffers are created
static void cb_commit_wait(ec_evt * ev, id<MTLCommandQueue> q) {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    evt_wait(ev, cb);
    [cb commit];
}

// gate for staging/ring reuse: returns an index with pending==0 (mirrors CUDA's dstage_done
// event-query walk); if the ring is fully in flight, bounded wait on the oldest pending slot.
static int gate_pick(int * pend, dispatch_semaphore_t * done, int n, int next) {
    int s = next;
    while (PEND_LOAD(&pend[s]) > 0) {
        s = (s + 1) % n;
        if (s == next) break;   // fully in flight: wait on the oldest below (rare path)
    }
    if (PEND_LOAD(&pend[s]) > 0) {
        for (int i = 0; i < n; i++) {
            if (PEND_LOAD(&pend[i]) > 0) { s = i; break; }
        }
        dispatch_semaphore_wait(done[s], DISPATCH_TIME_FOREVER);
    }
    return s;
}

static int hook_is_gpu_buffer(const ggml_tensor * t);   // fwd — used by lookup_or_create

// --- entry creation --------------------------------------------------------------------------

// the scheduler can split an allocation that exceeds the buffer type max size into a multi-buffer; only the
// sub-buffer holding the tensor has a Metal view, so resolve it before asking for a buffer id
static ggml_metal_buffer_t ec_metal_ctx(ggml_backend_buffer_t buffer, const ggml_tensor * t) {
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        buffer = ggml_backend_multi_buffer_get_sub_buffer(buffer, t->data);
    }
    return (ggml_metal_buffer_t) buffer->context;
}

// [async] quiesce speculative fills before destroying/rebuilding an entry (model reload rail).
static void ec_entry_quiesce(ggml_metal_ec_entry * e) {
    {
        std::lock_guard<std::mutex> lk(g_ec_mtx);
        __atomic_store_n(&e->alive, 0, __ATOMIC_RELEASE);     // no new claims
        for (auto it = g_ec_jobs.begin(); it != g_ec_jobs.end(); )
            it = (it->e == e) ? g_ec_jobs.erase(it) : it + 1;
    }
    while (__atomic_load_n(&e->in_pf, __ATOMIC_RELAXED) > 0)  // bounded: one pread
        sched_yield();
}

static ggml_metal_ec_entry * lookup_or_create(ggml_backend_t backend, const ggml_tensor * host, const ggml_tensor * dev) {
    auto it = g_entries.find(host);
    if (it != g_entries.end()) {
        ggml_metal_ec_entry * e = &it->second;
        // [stale-entry rail] the host tensor can be freed and its ADDRESS recycled by a later model
        // load (llama-cli loads twice: fit/probe then real). An old entry then serves a stale
        // host_base and an orphaned GPU arena — detect the recycling and rebuild from scratch.
        if (e->host_base != host->data || e->slab != host->nb[2] || e->n_expert != host->ne[2]) {
            fprintf(stderr, "[metal-ec] STALE entry %s — host moved (%p -> %p), rebuilding\n",
                    ggml_get_name(host), e->host_base, host->data);
            ec_entry_quiesce(e);   // [async] wait out speculative fills before yanking fd/arena
            if (e->fd >= 0) ::close(e->fd);
            if (e->arena) [e->arena release];
            // staging is global now; nothing per-entry to release
            for (int r = 0; r < EC_ID_RING; r++) {
                if (e->ring_dst[r]) [e->ring_dst[r] release];
                if (e->ring_src[r]) [e->ring_src[r] release];
                free(e->ring_host[r]);
            }
            if (e->copyQ) [e->copyQ release];
            g_entries.erase(it);
        } else {
            // this call's device-side copy is (possibly) a different sched slot — re-index it
            e->dev_ptr = dev;
            g_dev_idx[dev] = e;
            return e;
        }
    }
    GGML_ASSERT(host->buffer && !hook_is_gpu_buffer(host)); // entry mirrors HOST weights, not device

    ggml_metal_ec_entry e;
    e.host_ptr  = host;
    e.dev_ptr   = dev;
    e.host_base = host->data;          // GGUF mmap base (slow, cold fallback — pages fault from file)
    e.slab      = host->nb[2];         // fused expert slab bytes: identical to CUDA
    e.n_expert  = host->ne[2];
    __atomic_store_n(&e.alive, 1, __ATOMIC_RELEASE);          // [async] (re)enable after build
    __atomic_store_n(&e.in_pf, 0, __ATOMIC_RELAXED);
    { std::lock_guard<std::mutex> el(g_ec_entries_mtx); g_ec_entries.push_back(&e); }  // [async]
    { const char * nm = ggml_get_name(host); e.layer_idx = -1;
      if (nm) { const char * q = strstr(nm, "blk."); if (q) { int L = atoi(q+4); if (L >= 0 && L < EC_MAX_LAYERS) e.layer_idx = L; } } }
    e.C         = ec_n_cached();

    // Metal device resolved from the tensor's buffer type (device->context IS ggml_metal_device_t
    // in this tree — see ggml-metal.cpp), so no accessor into the private backend-context struct.
    ggml_backend_dev_t bdev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(dev->buffer));
    GGML_ASSERT(bdev && ggml_backend_dev_type(bdev) == GGML_BACKEND_DEVICE_TYPE_GPU);
    e.mdev = (ggml_metal_device_t) bdev->context;
    id<MTLDevice> device = (id<MTLDevice>) ggml_metal_device_get_obj(e.mdev);

    // [disk source] same registry as CUDA: private O_RDONLY fd + double-buffered staging.
    // Expert bytes never materialize in host RAM beyond the 2-slab staging ring — the bounded-RSS
    // product requirement on 24 GB unified-memory machines.
    {
        const char * src = getenv("LLAMA_EXPERT_CACHE_SOURCE");
        const bool want_disk = src && strcmp(src, "disk") == 0;
        ggml_ec_file_src fs;
        if (want_disk && host->name[0] && ggml_backend_ec_get_file_source(host->name, &fs)) {
            int fd = ::open(fs.path, O_RDONLY);
            if (fd < 0) {
                fprintf(stderr, "[metal-ec] disk source: open(%s) failed -> mapping fallback for %s\n",
                        fs.path, host->name);
            } else {
                e.disk      = true;
                e.fd        = fd;
                e.file_offs = fs.offs;
                static std::atomic<int> n_disk{0};
                if (n_disk.fetch_add(1) == 0) {
                    fprintf(stderr, "[metal-ec] disk source ACTIVE (pread %s @%llu, slab=%.1fMB)\n",
                            fs.path, (unsigned long long) fs.offs, e.slab / 1e6);
                }
            }
        }
    }

    // arena: when the scheduler created a bounded copy (ne[2] == C) for a decode-sized graph, that buffer IS
    // the arena - no second allocation. Zero once at adoption so block-quant padding is never NaN.
    const bool dbg_shared = getenv("LLAMA_EC_DEBUG_SLOT_VERIFY") != nullptr;   // diag: host-readable arena
    // in verify mode we never adopt the scheduler's private bounded copy — we need .contents
    // [isolation] LLAMA_EC_NO_ADOPT=1: production keeps its OWN private arena (like verify mode) and
    // only uses the bounded copy for routing/layout. If the garbage-token bug is adoption-related
    // (galloc churn over the adopted buffer), this flips clean-run text coherent at prod rails.
    const bool no_adopt  = dbg_shared || getenv("LLAMA_EC_NO_ADOPT") != nullptr;
    if (dev->ne[2] == e.C && dev->buffer && !no_adopt) {
        struct ggml_metal_buffer_id bid = ggml_metal_buffer_get_id(ec_metal_ctx(dev->buffer, dev), dev);
        if (bid.metal) {
            e.arena     = (id<MTLBuffer>) bid.metal;
            e.arena_off = bid.offs;
            e.bounded   = true;
            [e.arena retain];
        } else {
            // bounded copy landed on a non-Metal buffer (mapped/CPU): keep the entry's own private arena, which
            // is what the intercept reads for served calls
            e.arena = [device newBufferWithLength:(size_t) e.C * e.slab
                                          options:dbg_shared ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate];
            [e.arena retain];
            e.own_arena = true;
        }
    } else {
        e.arena = [device newBufferWithLength:(size_t) e.C * e.slab
                                      options:dbg_shared ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate];
        [e.arena retain];
        e.own_arena = true;
    }
    {
        id<MTLCommandQueue> q = (id<MTLCommandQueue>) ggml_metal_device_get_queue(e.mdev);
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
        [enc fillBuffer:e.arena range:NSMakeRange(e.arena_off, (size_t) e.C * e.slab) value:0];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];   // load-time only
    }

    // [perf] staging ring is global (ec_stage_ring_ensure) — created on first fill, sized to the
    // largest slab. No per-entry allocation.

    // dedicated copy command queue + fork event
    e.copyQ = [device newCommandQueue];
    [e.copyQ retain];
    e.fork_ev = evt_init(device);

    e.slots.assign(e.C, ggml_metal_ec_slot{});
    e.slot_of_id.assign(e.n_expert, -1);

    // slot-ids ring: host-mapped shared sources (contents mapped once) + private destinations
    for (int r = 0; r < EC_ID_RING; r++) {
        int32_t * arr = (int32_t *) calloc(EC_RING_CAP_INTS, sizeof(int32_t));
        e.ring_host[r] = arr;
        e.ring_src[r] = [device newBufferWithBytesNoCopy:arr length:EC_RING_CAP_INTS * sizeof(int32_t)
                                                 options:MTLResourceStorageModeShared deallocator:nil];
        [e.ring_src[r] retain];
        e.ring_dst[r] = [device newBufferWithLength:EC_RING_CAP_INTS * sizeof(int32_t)
                                             options:MTLResourceStorageModePrivate];
        [e.ring_dst[r] retain];
        e.ring_pending[r] = 0;
        e.ring_done[r] = dispatch_semaphore_create(0);
    }

    auto res = g_entries.emplace(host, std::move(e));
    g_dev_idx[dev] = &res.first->second;
    return &res.first->second;
}

// --- hooks called from the ggml-backend.cpp sched copy phase ---------------------------------

static int hook_active(void) {
    return ec_n_cached() > 0;
}

static int hook_is_gpu_buffer(const ggml_tensor * t) {
    // memoized buffer verdict (mirrors CUDA's M3-F hot-path optimization)
    static std::unordered_map<ggml_backend_buffer_t, char> memo;
    const ggml_backend_buffer_t buf = t->buffer;
    if (!buf) return 0;
    auto it = memo.find(buf);
    if (it != memo.end()) return it->second;
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buf));
    const char r = (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) ? 1 : 0;
    memo.emplace(buf, r);
    return r;
}

static int hook_disk(const ggml_tensor * host) {
    auto it = g_entries.find(host);
    // g_entries stores entry VALUES (g_dev_idx stores pointers) — dot access here
    return (it != g_entries.end() && it->second.disk && it->second.fd >= 0) ? 1 : 0;
}

static int hook_lookup_or_evict(const ggml_tensor * dev, int32_t eid, int64_t * out_slot, int * out_need_copy);
static int ec_copy_to_slot_impl(ggml_metal_ec_entry * e, int32_t eid, int64_t slot);   // fwd (defined below)

// [policy] per-layer top-k pins + prewarm. g_ec_freq accumulates routes (shared across runs via
// LLAMA_EC_PROFILE). At first setup for an entry: pread the top-k experts into slots and mark hot
// (the existing 3-pass victim walk protects hot slots first). Every 64 gens the pin set is
// recomputed from counts; slots outside it lose hot and re-enter LRU. Routing-transparent: only
// residency order changes, never which expert the ids ring points at.
static void ec_repin(ggml_metal_ec_entry * e, int do_prewarm) {
    const int k = g_ec_pin_k > 0 ? g_ec_pin_k : (int) (e->C / 3);
    if (e->layer_idx < 0 || k <= 0) return;
    uint32_t top_e[EC_N_EXPERT_MAX]; uint32_t top_v[EC_N_EXPERT_MAX];
    int nk = 0;
    for (int cand = 0; cand < (int) e->n_expert; cand++) {
        uint32_t v = g_ec_freq[e->layer_idx*EC_N_EXPERT_MAX + cand].load(std::memory_order_relaxed);
        if (!v && !do_prewarm) continue;
        int j = nk;
        while (j > 0 && top_v[j-1] < v) { top_v[j] = top_v[j-1]; top_e[j] = top_e[j-1]; j--; }
        if (nk < k) { nk++; }
        top_v[j] = v; top_e[j] = cand;
    }
    // drop zero-freq candidates unless prewarming from a profile with data
    while (nk > 0 && !top_v[nk-1]) nk--;
    if (!nk) return;
    bool is_pin[EC_N_EXPERT_MAX] = {};
    for (int i = 0; i < nk; i++) is_pin[top_e[i]] = true;
    for (int64_t s = 0; s < e->C; s++) {
        const int32_t id = e->slots[s].expert_id;
        if (id >= 0) e->slots[s].hot = is_pin[id];
    }
    if (!do_prewarm) return;
    for (int i = 0; i < nk; i++) {
        const int32_t eid = (int32_t) top_e[i];
        if (e->slot_of_id[eid] >= 0) { e->slots[e->slot_of_id[eid]].hot = true; continue; }
        int64_t s = -1; int need = 0;
        if (!hook_lookup_or_evict(e->dev_ptr, eid, &s, &need) || !need) continue;
        if (ec_copy_to_slot_impl(e, eid, s)) e->slots[s].hot = true;
    }
    e->prewarmed = 1;
}

static int hook_setup(ggml_backend_t backend, const ggml_tensor * host, ggml_tensor * dev, int64_t n_tokens, int64_t n_ids_elems) {
    ggml_metal_ec_entry * e = lookup_or_create(backend, host, dev);
    if (!e) return 0;   // diagnostic decline
    e->gen++;
    e->ids_expect   = n_ids_elems;
    e->partial_fill = 0;
    // engage ONLY for decode-sized ubatches (mirrors CUDA; prefill keeps the legacy path byte-identical)
    e->engaged = (n_tokens <= 8 && e->C >= n_ids_elems) ? 1 : 0;
    if (e->engaged) ec_start_worker_once();   // [async] lazy start: only when serving begins
    if (!e->prewarmed && e->gen >= 2) ec_repin(e, /*do_prewarm=*/1);   // once; after the first prompt
    e->last_ring_idx = -1;
    e->last_ids_n    = 0;
    g_ec_setups++;
    return 1;
}

// LRU+stamp eviction, 3-pass — ported verbatim from expert-cache.cu (pure CPU logic)
static int hook_lookup_or_evict(const ggml_tensor * dev, int32_t eid, int64_t * out_slot, int * out_need_copy) {
    auto it = g_dev_idx.find(dev);
    GGML_ASSERT(it != g_dev_idx.end());
    ggml_metal_ec_entry * e = it->second;   // value IS a pointer here (g_dev_idx stores entry*)
    GGML_ASSERT(eid >= 0 && eid < e->n_expert);

    // [async] hit lookup under the metadata lock. A worker may own the mapping (state LOADING,
    // loading_eid == eid): then WAIT on its read instead of duplicating the pread — a bounded
    // one-pread wait, never worse than reading it ourselves, and a prefetch that lands in time
    // for demand converts a would-be miss into a counted hit.
    if (g_async_on) {
        std::unique_lock<std::mutex> lk(g_ec_mtx);
        const int64_t s0 = e->slot_of_id[eid];
        if (s0 >= 0) {
            ggml_metal_ec_slot & sl0 = e->slots[s0];
            if (sl0.state == EC_ST_READY && sl0.expert_id == eid) {
                sl0.stamp_gen = e->gen;
                sl0.last_use  = e->gen;
                if (sl0.was_prefetched) { g_ec_pf_hit++; sl0.was_prefetched = false; }
                *out_slot = s0; *out_need_copy = 0;
                e->partial_fill = 1;
                g_ec_hits++; g_ec_calls++;
                return 1;
            }
            if (sl0.state == EC_ST_LOADING && sl0.loading_eid == eid) {
                g_ec_cv.wait_for(lk, std::chrono::milliseconds(200), [&] {
                    return g_ec_quit.load() || e->slot_of_id[eid] != s0 ||
                           e->slots[s0].state == EC_ST_READY ||
                           (e->slots[s0].state == EC_ST_LOADING && e->slots[s0].loading_eid != eid);
                });
                const int64_t s1 = e->slot_of_id[eid];
                if (s1 >= 0 && e->slots[s1].state == EC_ST_READY && e->slots[s1].expert_id == eid) {
                    ggml_metal_ec_slot & sl1 = e->slots[s1];
                    sl1.stamp_gen = e->gen;
                    sl1.last_use  = e->gen;
                    if (sl1.was_prefetched) { g_ec_pf_hit++; sl1.was_prefetched = false; }
                    *out_slot = s1; *out_need_copy = 0;
                    e->partial_fill = 1;
                    g_ec_hits++; g_ec_calls++;
                    return 1;
                }
                if (s1 >= 0 && e->slots[s1].state == EC_ST_LOADING && e->slots[s1].loading_eid == eid) {
                    // owner stalled past the wait (or died mid-read): steal the claim so demand can
                    // read it now. The owner's settle validates its claim and will no-op after.
                    e->slot_of_id[eid] = -1;
                    e->slots[s1].state       = EC_ST_FREE;
                    e->slots[s1].expert_id   = -1;
                    e->slots[s1].loading_eid = -1;
                }
                // fall through to normal victim selection
            }
        }
    }
    if (e->slot_of_id[eid] >= 0) {
        int64_t s = e->slot_of_id[eid];
        if (g_async_on && !(e->slots[s].state == EC_ST_READY && e->slots[s].expert_id == eid)) {
            // stale mapping (claim/settle interleaving): drop it and re-claim below
            e->slot_of_id[eid] = -1;
        } else {
            e->slots[s].stamp_gen = e->gen;      // touched this call: never evict
            e->slots[s].last_use  = e->gen;
            *out_slot      = s;
            *out_need_copy = 0;
            e->partial_fill = 1;                 // legacy full copy was skipped for this slab
            g_ec_hits++;
            g_ec_calls++;
            return 1;
        }
    }

    // victim: empty slot > LRU-oldest unstamped > LRU-oldest (never stamp this call's own ids).
    // hot slots evicted only after every cold candidate is exhausted (preheat deferred to v1.1,
    // so pass 0 degenerates); in-flight copies are safe to overwrite (queue ordering), but prefer
    // finished ones — the same three passes as CUDA.
    std::unique_ptr<std::lock_guard<std::mutex>> vlk;   // [async] claim atomic vs the worker
    if (g_async_on) vlk.reset(new std::lock_guard<std::mutex>(g_ec_mtx));
    if (e->slot_of_id[eid] >= 0 && e->slots[e->slot_of_id[eid]].state == EC_ST_READY &&
        e->slots[e->slot_of_id[eid]].expert_id == eid) {
        // became resident while we were unlocked between the hit probe and here: take the hit
        const int64_t sh = e->slot_of_id[eid];
        e->slots[sh].stamp_gen = e->gen;
        e->slots[sh].last_use  = e->gen;
        if (e->slots[sh].was_prefetched) { g_ec_pf_hit++; e->slots[sh].was_prefetched = false; }
        *out_slot = sh; *out_need_copy = 0;
        e->partial_fill = 1;
        g_ec_hits++; g_ec_calls++;
        vlk.reset();
        return 1;
    }
    int64_t victim = -1;
    for (int64_t s = 0; s < e->C; s++) {
        // fresh never-used slots are FREE by design: open to demand AND worker. Only LOADING is owned.
        if (g_async_on && e->slots[s].state == EC_ST_LOADING) continue;
        if (e->slots[s].expert_id < 0) { victim = s; break; }
    }
    if (victim < 0) {
        for (int pass = 0; pass < 3 && victim < 0; pass++) {
            uint64_t oldest = UINT64_MAX;
            for (int64_t s = 0; s < e->C; s++) {
                const ggml_metal_ec_slot & sl = e->slots[s];
                if (g_async_on && sl.state == EC_ST_LOADING) continue;  // [async] owned by a reader
                if (sl.stamp_gen == e->gen) continue;          // in use by this call
                if (sl.hot && pass == 0) continue;
                const bool inflight = sl.in_flight && evt_in_flight(sl.in_flight);
                if (inflight && pass < 2 && oldest != UINT64_MAX) continue;  // prefer settled slots
                if (sl.last_use < oldest) { oldest = sl.last_use; victim = s; }
            }
        }
    }
    GGML_ASSERT(victim >= 0);   // only possible if EVERY slot is stamp==gen: C < routed width

    ggml_metal_ec_slot & sl = e->slots[victim];
    const int32_t evicted = sl.expert_id;
    if (sl.expert_id >= 0) {
        e->slot_of_id[sl.expert_id] = -1;
        g_ec_evicts++;
    }
    sl.expert_id   = eid;
    sl.hot         = false;     // fresh occupants are cold; only preheat marks hot (deferred v1.1)
    sl.stamp_gen   = e->gen;
    sl.last_use    = e->gen;
    if (g_async_on) { sl.state = EC_ST_LOADING; sl.loading_eid = eid; sl.was_prefetched = false; }
    e->slot_of_id[eid] = victim;
    *out_slot      = victim;
    *out_need_copy = 1;
    e->partial_fill = 1;
    g_ec_misses++;
    g_ec_calls++;
    vlk.reset();   // claim recorded; the pread itself runs unlocked in the caller
    // [async] the expert we just evicted is the single most likely near-future re-demand (skewed
    // routing, reuse distance ~W>C): queue it for the speculative worker, which will pread it back
    // into some free slot during the GPU window. Advisory: a PF job whose eid got resolved gets
    // dropped at fill time.
    if (g_async_on && evicted >= 0) ec_pf_enqueue(e, evicted);
    return 1;
}

// miss fill: pread slab into the next free staging buffer (host blocks only on the NVMe read),
// then blit onto copyQ ordered behind the fork event recorded on the device queue at issue time.
static const ggml_tensor * host_nm_guard(const ggml_metal_ec_entry * e) { return e->host_ptr; }
struct ec_timer2 {
    std::atomic<uint64_t> * d; std::atomic<uint64_t> * n; uint64_t t0;
    ec_timer2(std::atomic<uint64_t> & dd, std::atomic<uint64_t> & nn) : d(&dd), n(&nn) { t0 = ec_now_ns(); }
    ~ec_timer2() { d->fetch_add(ec_now_ns() - t0, std::memory_order_relaxed); n->fetch_add(1, std::memory_order_relaxed); }
};

static int ec_copy_to_slot_impl(ggml_metal_ec_entry * e, int32_t eid, int64_t slot) {
    GGML_ASSERT(e && slot >= 0 && slot < e->C);
    typedef ec_timer2 ec_timer;
    static const bool perf_on = getenv("LLAMA_EC_PERF") != nullptr;   // cached: getenv per fill is a scan
    std::unique_ptr<ec_timer> t_prep, t_pread, t_blit;
    if (perf_on) t_prep.reset(new ec_timer(g_perf.gate_ns, g_perf.gate_n));
    if (getenv("LLAMA_EC_DEBUG_SLOT_VERIFY") && e->dbg_trace < 64) {
        e->dbg_trace++;
        fprintf(stderr, "[ec-fill] %s gen=%llu eid=%d slot=%lld was_res=%d\n", ggml_get_name(e->host_ptr),
                (unsigned long long) e->gen, eid, (long long) slot, (int) (e->slot_of_id[eid] >= 0));
    }

    ec_stage_ring_ensure((id<MTLDevice>) ggml_metal_device_get_obj(e->mdev), e->slab);
    int s = gate_pick(g_stage_pend, g_stage_done, EC_STAGE_RING, g_stage_next);
    g_stage_next = (s + 1) % EC_STAGE_RING;
    t_prep.reset();

    bool from_staging = false;
    // [Mac-native fill] Host-readable arena (storageModeShared — what the adopted scheduler copy
    // actually is on this build): pread STRAIGHT into the slot bytes. No staging ring, no memcpy,
    // no blit, no fork/join, no command buffer. Correct because fills run in the scheduler copy
    // phase, strictly before the graph's command buffer is committed, and shared stores are coherent
    // with kernels in buffers committed after the store (Apple Silicon). The slot is never rebound
    // while its old gen is in flight (victim rule), so no reader can observe a half-written slab.
    if (e->disk && e->fd >= 0 && e->arena.contents) {
        if (perf_on) t_pread.reset(new ec_timer(g_perf.pread_ns, g_perf.pread_n));
        uint8_t * dst = (uint8_t *) e->arena.contents + e->arena_off + (size_t) slot * e->slab;
        ssize_t got = ::pread(e->fd, dst, e->slab,
                              (off_t) (e->file_offs + (uint64_t) eid * e->slab));
        if (got == (ssize_t) e->slab) {
            ggml_metal_ec_slot & sl0 = e->slots[slot];
            sl0.expert_id = eid;
            sl0.loading = false;                       // bytes already in place (synchronous fill)
            if (g_async_on) { sl0.state = EC_ST_READY; sl0.loading_eid = -1; g_ec_cv.notify_all(); }
            g_ec_copy_bytes += e->slab;
            g_ec_side_copies++;
            return 1;
        }
        fprintf(stderr, "[metal-ec] disk pread short (%zd/%zu) eid=%d -> staging fallback\n",
                got, e->slab, eid);
        if (g_async_on) {
            // hand the claim back so later demand can retry cleanly
            ggml_metal_ec_slot & slf = e->slots[slot];
            if (slf.state == EC_ST_LOADING && slf.loading_eid == eid) {
                e->slot_of_id[eid] = -1;
                slf.state = EC_ST_FREE; slf.expert_id = -1; slf.loading_eid = -1;
                g_ec_cv.notify_all();
            }
        }
    }
    if (e->disk && e->fd >= 0 && g_stage[s] && !e->arena.contents) {
        if (perf_on) t_pread.reset(new ec_timer(g_perf.pread_ns, g_perf.pread_n));
        ssize_t got = ::pread(e->fd, g_stage[s].contents, e->slab,
                              (off_t) (e->file_offs + (uint64_t) eid * e->slab));
        if (got != (ssize_t) e->slab) {
            fprintf(stderr, "[metal-ec] disk pread short (%zd/%zu) eid=%d -> mapping fallback\n",
                    got, e->slab, eid);
        } else {
            from_staging = true;
        }
    }
    if (!from_staging) {
        // mapping fallback: host_base pages demand-fault from the same file (slow but correct).
        // Blit encoders cannot read a raw host pointer, so copy into staging first.
        if (!g_stage[s]) return 0;   // ring allocation failed: decline
        memcpy(g_stage[s].contents, (const uint8_t *) e->host_base + (size_t) eid * e->slab, e->slab);
        from_staging = true;
    }

    // fork: order the copy behind everything currently on the device queue (same contract as
    // cudaEventRecord on the kernel stream at issue time — a slot may still be read by an earlier
    // graph's kernels)
    id<MTLCommandQueue> dq = device_queue(e);
    // [race isolation] LLAMA_EC_FILL_ON_DQ=1: fill blits FIFO on the DEVICE queue instead of the
    // side copyQ + fork event. fill_ids (and thus the graph's command buffers) is always committed
    // later on the same queue, so FIFO ordering subsumes the join. A/B against copyQ to find which
    // half of the CUDA-mirror ordering is broken on Metal.
    const bool fill_on_dq = getenv("LLAMA_EC_FILL_ON_DQ") != nullptr;
    if (perf_on) { t_prep.reset(); t_blit.reset(new ec_timer(g_perf.blit_ns, g_perf.blit_n)); }
    cb_commit_signal(e->fork_ev, dq);

    id<MTLCommandBuffer> cb = fill_on_dq ? [dq commandBuffer] : [e->copyQ commandBuffer];
    if (!fill_on_dq) evt_wait(e->fork_ev, cb);
    id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
        [enc copyFromBuffer:g_stage[s] sourceOffset:0 toBuffer:e->arena
              destinationOffset:e->arena_off + (size_t) slot * e->slab size:e->slab];
    [enc endEncoding];

    ggml_metal_ec_slot & sl = e->slots[slot];
    if (!sl.in_flight) sl.in_flight = evt_init((id<MTLDevice>) ggml_metal_device_get_obj(e->mdev));
    evt_signal(sl.in_flight, cb);
    PEND_ADD(&g_stage_pend[s], 1);
    dispatch_semaphore_t sdone = g_stage_done[s];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> b){ GGML_UNUSED(b); PEND_ADD(&g_stage_pend[s], -1); dispatch_semaphore_signal(sdone); }];

    [cb commit];
    sl.loading = true;   // cleared when fill_ids joins the device queue onto this slot
    g_ec_copy_bytes += e->slab;
    g_ec_side_copies++;
    return 1;
}

static int hook_copy_to_slot(ggml_backend_t backend, const ggml_tensor * dev, int32_t eid, int64_t slot) {
    auto it = g_dev_idx.find(dev);
    if (it == g_dev_idx.end()) return 0;
    GGML_UNUSED(backend);   // device queue resolved via the entry's device
    return ec_copy_to_slot_impl(it->second, eid, slot);
}

// --- [async] worker + PF queue ------------------------------------------------------------
// [demand-parallel] post one already-claimed slot for a worker to read, then (barrier=1) wait
// for the whole wave to settle. Called from the scheduler copy phase BEFORE the graph commit, so
// slot bytes are complete before any kernel reads them — same lifetime contract as the serial
// path, just N-way parallel preads. A slot still LOADING after the wave (worker died / short
// read) is filled synchronously by the caller: correctness never depends on the pool.
static void ec_demand_post_wait(ggml_metal_ec_entry * e, int32_t eid, int64_t slot) {
    {
        std::lock_guard<std::mutex> lk(g_ec_mtx);
        // no PF clear (a full clear starved speculation: pf landed=7 and demand still paid its
        // full pread chain). Workers drain demand jobs first; PF reads interleave in the gaps.
        g_ec_djobs.push_back(ec_djob{e, eid, slot});
        g_ec_dpend++;                              // under the lock: barrier can't race the post
    }
    g_ec_cv.notify_one();
}

static void ec_demand_drain_barrier(void) {
    const uint64_t t0 = ec_now_ns();
    std::unique_lock<std::mutex> lk(g_ec_mtx);
    g_ec_dcv.wait_for(lk, std::chrono::milliseconds(500),
                      [] { return g_ec_quit.load() || g_ec_dpend.load() <= 0; });
    g_dbar_ns += ec_now_ns() - t0;
    g_dbar_n++;
}

static void ec_pf_enqueue(ggml_metal_ec_entry * e, int32_t eid) {
    if (!g_async_on || eid < 0) return;
    {
        std::lock_guard<std::mutex> lk(g_ec_mtx);
        if (g_ec_jobs.size() >= 4096) g_ec_jobs.pop_front();     // bounded; drop oldest (advisory)
        g_ec_jobs.push_back(ec_job{e, eid});
    }
    g_ec_cv.notify_one();
}

// PF victim pick (caller holds g_ec_mtx): empty READY > oldest unstamped READY. Never touch:
// LOADING (owner mid-read), bound_unfinished (committed graph may still read it), hot (policy),
// stamped for the current call.
static bool ec_pf_claim_locked(ggml_metal_ec_entry * e, int32_t eid, int64_t * out) {
    if (e->slot_of_id[eid] >= 0) return false;
    int64_t victim = -1;
    for (int64_t s = 0; s < e->C; s++) {
        ggml_metal_ec_slot & sl = e->slots[s];
        if ((sl.state == EC_ST_READY || sl.state == EC_ST_FREE) && sl.expert_id < 0 &&
            !sl.bound_unfinished && !sl.hot) { victim = s; break; }     // never-used: best victim
    }
    if (victim < 0) {
        uint64_t oldest = UINT64_MAX;
        for (int64_t s = 0; s < e->C; s++) {
            ggml_metal_ec_slot & sl = e->slots[s];
            if (sl.state == EC_ST_LOADING) continue;
            if (sl.bound_unfinished || sl.hot || sl.stamp_gen == e->gen) continue;
            if (sl.last_use < oldest) { oldest = sl.last_use; victim = s; }
        }
    }
    if (victim < 0) return false;
    ggml_metal_ec_slot & sl = e->slots[victim];
    if (sl.expert_id >= 0) e->slot_of_id[sl.expert_id] = -1;
    sl.state       = EC_ST_LOADING;
    sl.loading_eid = eid;
    sl.expert_id   = -1;
    sl.hot         = false;
    e->slot_of_id[eid] = victim;
    *out = victim;
    return true;
}

// [demand-parallel] worker reads bytes into a slot demand already reserved (state LOADING,
// loading_eid=eid, mapping set by hook_lookup_or_evict). The slot is NOT touched here beyond the
// byte write — demand owns the settle (READY or rollback), so the graph-visible contract stays
// exactly as before. A dead/foreign claim is skipped: demand's own pread fallback covers it.
static void ec_demand_fill_one(ggml_metal_ec_entry * e, int32_t eid, int64_t s) {
    if (!e || eid < 0 || eid >= (int32_t) e->n_expert || s < 0 || s >= e->C) { g_ec_dpend--; g_ec_dcv.notify_all(); return; }
    bool mine = false;
    {
        std::lock_guard<std::mutex> lk(g_ec_mtx);
        ggml_metal_ec_slot & sl = e->slots[s];
        mine = __atomic_load_n(&e->alive, __ATOMIC_ACQUIRE) &&
               e->slot_of_id[eid] == s && sl.state == EC_ST_LOADING && sl.loading_eid == eid;
        if (mine) __atomic_add_fetch(&e->in_pf, 1, __ATOMIC_RELAXED);
    }
    if (mine && e->disk && e->fd >= 0 && e->arena && e->arena.contents &&
        e->host_ptr->data == e->host_base) {
        uint8_t * dst = (uint8_t *) e->arena.contents + e->arena_off + (size_t) s * e->slab;
        ssize_t got = ::pread(e->fd, dst, e->slab, (off_t) (e->file_offs + (uint64_t) eid * e->slab));
        if (got == (ssize_t) e->slab) {
            std::lock_guard<std::mutex> lk(g_ec_mtx);
            ggml_metal_ec_slot & sl = e->slots[s];
            // settle as READY right now: demand's post-barrier check converts it to a hit path
            if (sl.state == EC_ST_LOADING && sl.loading_eid == eid && e->slot_of_id[eid] == s) {
                sl.expert_id   = eid;
                sl.state       = EC_ST_READY;
                sl.loading_eid = -1;
                sl.last_use    = e->gen;                 // stamped at fill: safe to recycle later
                g_ec_copy_bytes += e->slab;
                g_ec_side_copies++;
            }
        }
        // short read: leave LOADING; demand re-checks and does its own (fallback) pread
        std::lock_guard<std::mutex> lk2(g_ec_mtx);
        __atomic_sub_fetch(&e->in_pf, 1, __ATOMIC_RELAXED);
    } else if (mine) {
        std::lock_guard<std::mutex> lk(g_ec_mtx);
        __atomic_sub_fetch(&e->in_pf, 1, __ATOMIC_RELAXED);
    }
    g_ec_dpend--;
    g_ec_dcv.notify_all();
}

static bool ec_pf_fill_one(ggml_metal_ec_entry * e, int32_t eid) {
    if (!e || eid < 0 || eid >= (int32_t) e->n_expert) return false;
    if (!e->disk || e->fd < 0 || !e->arena || !e->arena.contents) return false;   // direct-fill only
    if (e->host_ptr->data != e->host_base) return false;                          // stale-entry rail
    int64_t s = -1;
    {
        std::lock_guard<std::mutex> lk(g_ec_mtx);
        if (!__atomic_load_n(&e->alive, __ATOMIC_ACQUIRE) ||
            !ec_pf_claim_locked(e, eid, &s)) { g_ec_pf_drop++; return false; }
        __atomic_add_fetch(&e->in_pf, 1, __ATOMIC_RELAXED);   // teardown waits for us
    }
    uint8_t * dst = (uint8_t *) e->arena.contents + e->arena_off + (size_t) s * e->slab;
    ssize_t got = ::pread(e->fd, dst, e->slab, (off_t) (e->file_offs + (uint64_t) eid * e->slab));
    bool ok = (got == (ssize_t) e->slab);
    std::lock_guard<std::mutex> lk(g_ec_mtx);
    __atomic_sub_fetch(&e->in_pf, 1, __ATOMIC_RELAXED);
    ggml_metal_ec_slot & sl = e->slots[s];
    if (sl.state == EC_ST_LOADING && sl.loading_eid == eid) {           // still ours?
        if (ok && e->slot_of_id[eid] == s) {
            sl.expert_id      = eid;
            sl.state          = EC_ST_READY;
            sl.loading_eid    = -1;
            sl.was_prefetched = true;
            sl.last_use       = e->gen;
            g_ec_pf_fill++;
            g_ec_copy_bytes += e->slab;
        } else {                                                        // roll our claim back
            if (e->slot_of_id[eid] == s) e->slot_of_id[eid] = -1;
            sl.state = EC_ST_FREE; sl.expert_id = -1; sl.loading_eid = -1;
            g_ec_pf_drop++; ok = false;
        }
        g_ec_cv.notify_all();
    } else {
        g_ec_pf_drop++;   // demand stole the slot while we read: our bytes are orphaned, harmless
    }
    return ok;
}

static void ec_start_worker_once(void) {
    static bool started = false;
    if (started || !g_async_on) return;
    started = true;
    const char * wn = getenv("LLAMA_EC_WORKERS");
    if (wn) g_nworkers = atoi(wn);
    if (g_nworkers < 1) g_nworkers = 1;
    if (g_nworkers > 16) g_nworkers = 16;
    for (int i = 0; i < g_nworkers; i++) {
        pthread_t t;
        pthread_create(&t, nullptr, ec_worker_thread, nullptr);
        pthread_detach(t);
    }
    fprintf(stderr, "[metal-ec] %d fill worker(s) online\n", g_nworkers);
}

static void * ec_worker_thread(void *) {
    for (;;) {
        ec_job       j{nullptr, 0};
        ec_djob      dj{nullptr, 0, -1};
        bool         have_d = false;
        {
            std::unique_lock<std::mutex> lk(g_ec_mtx);
            g_ec_cv.wait(lk, [] { return g_ec_quit.load() || !g_ec_jobs.empty() || !g_ec_djobs.empty(); });
            if (g_ec_quit.load()) return nullptr;
            if (!g_ec_djobs.empty())     { dj  = g_ec_djobs.front();  g_ec_djobs.pop_front();  have_d = true; }
            else if (!g_ec_jobs.empty()) { j   = g_ec_jobs.front();   g_ec_jobs.pop_front(); }
        }
        if (have_d) ec_demand_fill_one(dj.e, dj.eid, dj.slot);
        else        ec_pf_fill_one(j.e, j.eid);
    }
}

// [disk source] scheduler legacy-copy replacement: fill input_cpy slab [first_id, last_id] from the
// GGUF through the staging ring ON THE DEVICE QUEUE — same ordering contract as the stock
// set_async-from-host it replaces (graph reads are ordered behind it on that queue).
static void ec_fill_dev_slabs(ggml_metal_ec_entry * e, const ggml_tensor * dev, int32_t first_id, int32_t last_id) {
    id<MTLCommandQueue> dq = device_queue(e);
    struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(ec_metal_ctx(dev->buffer, dev), dev);
    if (bid_dst.metal == nil) {
        // no Metal views on this buffer (mapped/CPU): blit cannot target it; caller must serve from the arena
        GGML_LOG_ERROR("%s: refusing blit into non-Metal buffer for %s\n", __func__, ggml_get_name(dev));
        return;
    }
    ec_stage_ring_ensure((id<MTLDevice>) ggml_metal_device_get_obj(e->mdev), e->slab);
    for (int32_t eid = first_id; eid <= last_id; eid++) {
        int s = gate_pick(g_stage_pend, g_stage_done, EC_STAGE_RING, g_stage_next);
        g_stage_next = (s + 1) % EC_STAGE_RING;
        if (!g_stage[s]) continue;   // staging failed at create: caller declined via hook_disk

        ssize_t got = ::pread(e->fd, g_stage[s].contents, e->slab,
                              (off_t) (e->file_offs + (uint64_t) eid * e->slab));
        if (got != (ssize_t) e->slab) continue;   // short read: same policy as CUDA

        id<MTLCommandBuffer> cb = [dq commandBuffer];
        id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
        [enc copyFromBuffer:g_stage[s] sourceOffset:0 toBuffer:(id<MTLBuffer>) bid_dst.metal
              destinationOffset:bid_dst.offs + (size_t) eid * e->slab size:e->slab];
        [enc endEncoding];
        PEND_ADD(&g_stage_pend[s], 1);
        dispatch_semaphore_t sdone = g_stage_done[s];
        [cb addCompletedHandler:^(id<MTLCommandBuffer> b){ GGML_UNUSED(b); PEND_ADD(&g_stage_pend[s], -1); dispatch_semaphore_signal(sdone); }];
        [cb commit];
    }
}

static void hook_copy_slabs(ggml_backend_t backend, ggml_tensor * dev, int32_t first_id, int32_t last_id) {
    auto it = g_dev_idx.find(dev);
    if (it == g_dev_idx.end() || !it->second->disk || it->second->fd < 0) return;   // defensive: caller only routes here when disk(host) said yes
    GGML_UNUSED(backend);
    ggml_metal_ec_entry * e = it->second;
    // [debug rail] in slot-verify mode the scheduler copy is only C slabs; ec_fill_dev_slabs would
    // write eid*slab (eid up to 127) into it = GPU OOB writes into neighbouring arenas. Route every
    // fill through the entry slots instead (exactly what bounded entries already do). Same for
    // own-arena entries: the intercept never reads the scheduler copy, and with C<n_expert the
    // entry can only serve via slots.
    if (e->bounded || e->own_arena || getenv("LLAMA_EC_DEBUG_SLOT_VERIFY")) {
        // a bounded copy holds C slots only, so this call's ids must fit in the arena. Route them through the
        // entry slots instead of blitting into the tensor; a wider call cannot be served.
        if (!e->engaged || (last_id - first_id + 1) > e->C) return;
        static const bool dpar = g_async_on && [] {
            const char * d = getenv("LLAMA_EC_DEMAND_PAR");
            return !(d && atoi(d) == 0);
        }();
        // decode-only in practice: the bounded branch already requires e->engaged (set per call
        // from n_tokens<=8), and prefill MoE runs on CPU — this hook never fires for prompt.
        if (dpar) {
            // wave: claim every miss, post all reads to the pool, barrier, then settle what the
            // pool could not finish (sync fallback) — all still before the graph commit.
            struct ec_wf { int32_t eid; int64_t slot; };
            ec_wf wave[512];
            int   nw = 0;
            for (int32_t eid = first_id; eid <= last_id; eid++) {
                int64_t slot;
                int need_copy = 0;
                hook_lookup_or_evict(dev, eid, &slot, &need_copy);
                if (need_copy && nw < 512) wave[nw++] = ec_wf{ eid, slot };
            }
            g_dwave_n += nw;
            for (int i = 0; i < nw; i++) ec_demand_post_wait(e, wave[i].eid, wave[i].slot);
            if (nw) ec_demand_drain_barrier();
            for (int i = 0; i < nw; i++) {
                bool ready = false;
                {
                    std::lock_guard<std::mutex> lk(g_ec_mtx);
                    ggml_metal_ec_slot & sl = e->slots[wave[i].slot];
                    ready = sl.state == EC_ST_READY && sl.expert_id == wave[i].eid &&
                            e->slot_of_id[wave[i].eid] == wave[i].slot;
                }
                if (!ready) ec_copy_to_slot_impl(e, wave[i].eid, wave[i].slot);   // sync fallback
            }
            return;
        }
        for (int32_t eid = first_id; eid <= last_id; eid++) {
            int64_t slot;
            int need_copy = 0;
            hook_lookup_or_evict(dev, eid, &slot, &need_copy);
            if (need_copy) ec_copy_to_slot_impl(e, eid, slot);
        }
        return;
    }
    ec_fill_dev_slabs(e, dev, first_id, last_id);
}

// map this call's routed ids to slot indices and upload to the ring. Join (device queue behind
// pending copies) is committed HERE, before the graph encodes this call — fill_ids runs in the
// sched copy phase ahead of graph_compute, exactly where CUDA's join lives.
static int hook_fill_ids(ggml_backend_t backend, const ggml_tensor * dev, const int32_t * host_ids, int n) {
    static const bool perf_on2 = getenv("LLAMA_EC_PERF") != nullptr;
    std::unique_ptr<ec_timer2> t_all;
    if (perf_on2) t_all.reset(new ec_timer2(g_perf.fillids_ns, g_perf.fillids_n));
    auto it = g_dev_idx.find(dev);
    GGML_ASSERT(it != g_dev_idx.end());
    ggml_metal_ec_entry * e = it->second;
    // the arena holds C slots; a call that routes more ids than C cannot be served from it - decline instead
    if (!e->engaged || (int64_t) n > e->C) return 0;
    if ((int64_t) n > EC_RING_CAP_INTS) return 0;   // fixed-cap decline (grow path deferred to v1.1)
    // one layer per gen bucket (rotated) so a repin never stalls all 46 layers in one token
    if (g_ec_pin_k > 0 && e->prewarmed && (e->gen % 64) == 0 &&
        (e->layer_idx % 64) == (int) ((e->gen / 64) % 64) && (e->gen / 64) != e->last_repin) {
        e->last_repin = e->gen / 64;
        ec_repin(e, /*do_prewarm=*/0);
    }

    GGML_UNUSED(backend);
    id<MTLCommandQueue> dq = device_queue(e);

    int r = gate_pick(e->ring_pending, e->ring_done, EC_ID_RING, e->ring_next);
    e->ring_next = (r + 1) % EC_ID_RING;

    // AUTHORITATIVE resolution: every eid routed by THIS call is resolved here.
    // [garbage-token fix] PRE-STAMP PASS: mark every already-resident eid before any eviction can
    // run. Without it, the LRU walk below treats this call's own HIT slots as stale victims (their
    // stamp is from an earlier gen) and a later miss in the SAME call overwrites a slot the ring is
    // about to point at — slot_of[hit]=-1 while the kernel reads the slot, i.e. garbage MoE outputs.
    // [async] we are gen G+1's fill_ids: reaching here proves the logits for gen G were sampled
    // (a device-sync point), so gen G's kernels are retired — release their bindings so the worker
    // may recycle those slots.
    if (g_async_on) {
        std::lock_guard<std::mutex> lk(g_ec_mtx);
        for (int64_t s = 0; s < e->C; s++) e->slots[s].bound_unfinished = false;
    }
    int32_t * stage = e->ring_host[r];
    for (int i = 0; i < n; i++) {
        const int32_t eid = host_ids[i];
        if (eid < 0 || eid >= (int32_t) e->n_expert) continue;   // validated in the main loop below
        const int64_t s0 = e->slot_of_id[eid];
        if (s0 >= 0) {
            e->slots[s0].stamp_gen = e->gen;
            e->slots[s0].last_use  = e->gen;
        }
    }
    if (e->layer_idx >= 0 && e->layer_idx < EC_MAX_LAYERS) {
        for (int i = 0; i < n; i++) {
            const int32_t eid = host_ids[i];
            if (eid >= 0 && eid < e->n_expert)
                g_ec_freq[e->layer_idx*EC_N_EXPERT_MAX + eid].fetch_add(1, std::memory_order_relaxed);
        }
    }
    for (int i = 0; i < n; i++) {
        const int32_t eid = host_ids[i];
        if (eid < 0 || eid >= e->n_expert) {
            // scheduler fed routing outside [0, n_expert) — decline the cache path for this call;
            // fallback_full + legacy keeps output correct. Log once for forensics.
            static bool warned = false;
            if (!warned) {
                warned = true;
                fprintf(stderr, "[metal-ec] BAD id=%d at i=%d/%d (n_expert=%lld, gen=%llu) tensor=%s\n",
                        eid, i, n, (long long) e->n_expert, (unsigned long long) e->gen,
                        ggml_get_name(e->host_ptr));
            }
            e->engaged       = 0;
            e->last_ring_idx = -1;
            e->last_ids_n    = 0;
            return 0;
        }
        int64_t s = e->slot_of_id[eid];
        if (s < 0) {
            int need_copy = 0;
            hook_lookup_or_evict(dev, eid, &s, &need_copy);
            if (need_copy) hook_copy_to_slot(backend, dev, eid, s);
        } else if (perf_on2) {
            ec_timer2 t_hit(g_perf.hit_ns, g_perf.hit_n);
            stage[i] = (int32_t) s; continue;
        }
        stage[i] = (int32_t) s;
    }

    // [async] bind: gen G's kernels will read these slots; block worker recycling until gen G+1's
    // fill_ids clears the flags (the logits-sync proof above makes that safe).
    if (g_async_on) {
        std::lock_guard<std::mutex> lk(g_ec_mtx);
        for (int i = 0; i < n; i++) e->slots[stage[i]].bound_unfinished = true;
    }

    // join: the graph's command buffers must be ordered behind pending slot copies — including
    // slots that HIT this call but were mid-fill from an earlier miss. Dedupe on the slot.
    std::vector<int> dedup;
    for (int i = 0; i < n; i++) {
        const int64_t s = stage[i];
        ggml_metal_ec_slot & sl = e->slots[s];
        if (!sl.loading || !sl.in_flight) continue;
        bool dup = false;
        for (int j = 0; j < (int) dedup.size() && !dup; j++) dup = (dedup[j] == (int) s);
        if (dup) continue;
        dedup.push_back((int) s);
        // [garbage-token fix #2] ALWAYS commit the join wait for a loading slot. The CUDA heritage
        // skips it when cudaEventQuery says the copy is done — advisory there, and on Metal
        // evt_in_flight() has been observed to answer "settled" for blits still queued on copyQ
        // (clean8: correct bytes in slots, garbage tokens ~2 into decode = kernel read past an
        // unfinished fill). A spurious wait-only command buffer is cheap; a race is not.
        if (perf_on2) { ec_timer2 tj(g_perf.join_ns, g_perf.join_n); cb_commit_wait(sl.in_flight, dq); }
        else cb_commit_wait(sl.in_flight, dq);
        sl.loading = false;
    }

    // upload the ring entry: blit host-mapped source -> private destination on the device queue.
    // Ring reuse is queue-ordered behind any earlier graph still reading the older entry.
    id<MTLCommandBuffer> cb = [dq commandBuffer];
    id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
    [enc copyFromBuffer:e->ring_src[r] sourceOffset:0 toBuffer:e->ring_dst[r]
          destinationOffset:0 size:(size_t) n * sizeof(int32_t)];
    [enc endEncoding];
    PEND_ADD(&e->ring_pending[r], 1);
    dispatch_semaphore_t rdone = e->ring_done[r];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> b){ GGML_UNUSED(b); PEND_ADD(&e->ring_pending[r], -1); dispatch_semaphore_signal(rdone); }];
    [cb commit];

    // [debug] LLAMA_EC_DEBUG_SLOT_VERIFY: wait this call's copies, then memcmp every slot we are
    // about to serve against the authoritative mmap bytes. Splits copy-engine bugs (wrong bytes in
    // the slot) from intercept/binding bugs (right bytes, wrong kernel reads). Debug-only: shared
    // arena + blocking wait make it slow — never on for perf runs.
    if (getenv("LLAMA_EC_DEBUG_SLOT_VERIFY") && e->arena && e->arena.contents) {
        // LLAMA_EC_DEBUG_NOWAIT=1 isolates the join: keep shared-arena memcpy fills + audit but
        // SKIP the blocking wait, exactly like production. If text breaks with NOWAIT and stays
        // coherent without it, the bug is join/ordering; if it breaks either way, it's the fill.
        if (!getenv("LLAMA_EC_DEBUG_NOWAIT")) {
            id<MTLCommandBuffer> wcb = [dq commandBuffer];
            for (int i = 0; i < n; i++) {
                ggml_metal_ec_slot & sl = e->slots[stage[i]];
                if (sl.in_flight) evt_wait(sl.in_flight, wcb);
            }
            [wcb commit];
            [wcb waitUntilCompleted];
        }
        const uint8_t * ab = (const uint8_t *) e->arena.contents + e->arena_off;
        static int nv = 0;
        std::vector<uint8_t> filebuf;
        filebuf.resize((size_t) e->slab);
        for (int i = 0; i < n; i++) {
            const int32_t eid = host_ids[i];
            const int64_t s = stage[i];
            const uint8_t * got = ab + (size_t) s * e->slab;
            // authoritative: fresh pread from the same fd the copy engine uses
            ssize_t gr = ::pread(e->fd >= 0 ? e->fd : ::open("/dev/null", O_RDONLY),
                                 filebuf.data(), e->slab,
                                 (off_t) (e->file_offs + (uint64_t) eid * e->slab));
            if (gr != (ssize_t) e->slab) continue;
            if (memcmp(filebuf.data(), got, e->slab) != 0) {
                g_ec_verify_bad++;   // slot != file  -> COPY ENGINE / arena bug (real)
                size_t off = 0; while (off < e->slab && filebuf[off] == got[off]) off++;
                if (nv++ < 8) {
                    // scan the tensor's file range: which expert's bytes are ACTUALLY in the slot?
                    int match_eid = -2;
                    if (e->fd >= 0) {
                        static std::vector<uint8_t> scanbuf;
                        scanbuf.resize((size_t) e->slab);
                        for (int32_t cand = 0; cand < (int32_t) e->n_expert; cand++) {
                            ssize_t g2 = ::pread(e->fd, scanbuf.data(), e->slab,
                                                 (off_t) (e->file_offs + (uint64_t) cand * e->slab));
                            if (g2 == (ssize_t) e->slab && memcmp(scanbuf.data(), got, e->slab) == 0) {
                                match_eid = cand; break;
                            }
                        }
                    }
                    fprintf(stderr, "[ec-verify] SLOT!=FILE %s eid=%d slot=%lld first_diff=%zu slot_holds_eid=%d "
                                    "slot_of[%d]=%lld slot_of[%d]=%lld expert_id_in_slot=%d\n",
                            ggml_get_name(e->host_ptr), eid, (long long) s, off, match_eid,
                            eid, (long long) e->slot_of_id[eid],
                            match_eid >= 0 ? match_eid : 0,
                            match_eid >= 0 ? (long long) e->slot_of_id[match_eid] : -999,
                            (int) e->slots[s].expert_id);
                }
            } else {
                g_ec_verify_ok++;
            }
            // secondary: host mapping vs file (stale host_base diagnostic)
            const uint8_t * hb = (const uint8_t *) e->host_base + (size_t) eid * e->slab;
            if (memcmp(filebuf.data(), hb, e->slab) != 0) {
                g_ec_hoststale++;
                size_t off = 0; while (off < e->slab && filebuf[off] == hb[off]) off++;
                if (nv++ < 8) {
                    fprintf(stderr, "[ec-verify] HOST!=FILE %s eid=%d off=%zu (host=%02x file=%02x) [stale host_base?]\n",
                            ggml_get_name(e->host_ptr), eid, off, hb[off], filebuf[off]);
                }
            }
        }
    }

    if (getenv("LLAMA_EC_DEBUG_SLOT_VERIFY") && e->dbg_ids < 32) {
        e->dbg_ids++;
        fprintf(stderr, "[ec-ids] %s gen=%llu ids=[", ggml_get_name(e->host_ptr), (unsigned long long) e->gen);
        for (int i = 0; i < n; i++) fprintf(stderr, "%d%s", host_ids[i], i + 1 < n ? "," : "");
        fprintf(stderr, "] slots=[");
        for (int i = 0; i < n; i++) fprintf(stderr, "%lld%s", (long long) stage[i], i + 1 < n ? "," : "");
        fprintf(stderr, "]\n");
    }

    // [debug] deterministic post-fill audit: every slot we are about to serve must (a) be
    // bookkept to its eid and (b) hold the file bytes for that eid. One call before the ring is
    // published — no later-call interference possible.
    if (getenv("LLAMA_EC_DEBUG_SLOT_VERIFY") && e->arena && e->arena.contents) {
        const uint8_t * ab = (const uint8_t *) e->arena.contents + e->arena_off;
        std::vector<uint8_t> fb((size_t) e->slab);
        for (int i = 0; i < n; i++) {
            const int32_t eid = host_ids[i];
            const int64_t s = stage[i];
            const int bk = (int) e->slots[s].expert_id;
            const bool   bytes_ok = (e->fd >= 0 &&
                ::pread(e->fd, fb.data(), e->slab, (off_t) (e->file_offs + (uint64_t) eid * e->slab)) == (ssize_t) e->slab &&
                memcmp(fb.data(), ab + (size_t) s * e->slab, e->slab) == 0);
            if (!bytes_ok || bk != eid) {
                g_ec_audit_bad++;
                if (e->dbg_audit < 12) {
                    e->dbg_audit++;
                    const uint8_t * sp = ab + (size_t) s * e->slab;
                    bool all_zero = true;
                    for (size_t z = 0; z < e->slab; z += 4096) { if (sp[z]) { all_zero = false; break; } }
                    // cross-tensor contamination check: does the slot hold some OTHER slot region's bytes?
                    int other_slot = -1;
                    for (int64_t s2 = 0; s2 < e->C && s2 != s && other_slot < 0; s2++) {
                        if (memcmp(ab + (size_t) s2 * e->slab, sp, 256) == 0) other_slot = (int) s2;
                    }
                    ggml_ec_file_src fs_now;
                    bool reg_ok = e->host_ptr->name && ggml_backend_ec_get_file_source(e->host_ptr->name, &fs_now);
                    fprintf(stderr, "[ec-audit] %s gen=%llu i=%d eid=%d slot=%lld bk=%d bytes_ok=%d "
                                    "offs_entry=%llu offs_reg=%llu reg_ok=%d fd=%d disk=%d bounded=%d all_zero=%d dup_of_slot=%d b0=%02x\n",
                            ggml_get_name(e->host_ptr), (unsigned long long) e->gen, i, eid, (long long) s,
                            bk, (int) bytes_ok,
                            (unsigned long long) e->file_offs,
                            reg_ok ? (unsigned long long) fs_now.offs : 0ull, (int) reg_ok,
                            e->fd, (int) e->disk,
                            (int) e->bounded, (int) all_zero, other_slot, sp[0]);
                }
            } else {
                g_ec_audit_ok++;
            }
        }
    }

    e->last_ring_idx = r;
    e->last_ids_n    = n;
    g_ec_fillids++;
    return 1;
}

static int hook_served(const ggml_tensor * dev) {
    // true when the copy phase touched input_cpy for this call: if we then fail to intercept,
    // the legacy GEMM would read a partially-filled buffer -> fallback_full refill is mandatory
    // [blocker A] the intercept passes the ORIGINAL host tensor — resolve via host-key fallback.
    const ggml_metal_ec_entry * e = ec_entry_lookup(dev);
    return (e && e->partial_fill) ? 1 : 0;
}

static int hook_engaged(const ggml_tensor * dev) {
    ggml_metal_ec_entry * e = ec_entry_lookup(dev);   // [blocker A] host-key fallback
    if (!e) return 0;
    if (!e->engaged || e->last_ring_idx < 0) return 0;
    // the upload must match the ids tensor the kernel is about to read; any mismatch -> decline
    return (e->ids_expect == 0 || e->last_ids_n == e->ids_expect) ? 1 : 0;
}

static int64_t hook_slots(const ggml_tensor * dev) {
    const ggml_metal_ec_entry * e = ec_entry_lookup(dev);   // [blocker A] host-key fallback
    return e ? e->C : 0;
}

static const void * hook_arena(const ggml_tensor * dev) {
    // StorageModePrivate buffers have no host-visible base; the intercept binds Metal buffer ids
    // through ggml_metal_ec_arena_ref instead. NULL here is safe for any hook-table caller.
    GGML_UNUSED(dev);
    return nullptr;
}

static const int32_t * hook_ids_slot(const ggml_tensor * dev) {
    const ggml_metal_ec_entry * e = ec_entry_lookup(dev);   // [blocker A] host-key fallback
    if (!e || e->last_ring_idx < 0) return nullptr;
    return e->ring_host[e->last_ring_idx];   // host-mapped mirror of this call's entry
}

// [Metal EC] EC-managed weights must get a real device-side copy: the scheduler otherwise keeps them on the
// host buffer and reads them through mapped-buft support, which has no Metal views for kernels or blits.
static int hook_force_copy(const ggml_tensor * node, const ggml_tensor * host) {
    if (ec_n_cached() <= 0) return 0;
    // [gate75] only graphs the arena can serve may pull expert weights onto the GPU
    if (!ec_node_serve_eligible(node)) return 0;
    ggml_ec_file_src fs;
    return (host->name && host->name[0] && ggml_backend_ec_get_file_source(host->name, &fs)) ? 1 : 0;
}

// The device's own buffer type always has a Metal view; the mapped buft does not. EC-managed tensors are read
// by kernels and blits, so the scheduler must allocate them on this buft. CUDA answers NULL (unchanged path).
static ggml_backend_buffer_type_t hook_copy_buft(ggml_backend_t backend) {
    if (ec_n_cached() <= 0) return nullptr;
    return ggml_backend_get_default_buffer_type(backend);
}

// [Metal EC] bounded node-input allocation. The scheduler sizes this weight's copy buffer as C x slab when the
// graph is decode-sized and C can hold every routed id of the call; registry presence marks an EC-managed
// tensor. CUDA keeps full-size buffers by answering 0.
static int64_t hook_bounded_ne2(const ggml_tensor * node, const ggml_tensor * src, int64_t serve_max) {
    if (ec_n_cached() <= 0) return 0;
    // [gate73] STRICT decode-only: a bounded buffer is arena-shaped (C slabs) — a graph that
    // engages serves slots, but a graph that does NOT engage (prefill chunk, or elems>C at
    // prefill-time with C>=elems: ids ne[1]=2, elems=12) would get stock slab copies written at
    // full-size byte offsets into a C-slab buffer -> OOB GPU writes. Full-size copies are always
    // safe; the arena only pays off for graphs that are guaranteed to engage (ne[1]==1, C>=ne[0]).
    const ggml_tensor * ids = node->src[2];
    bool ids_ok = ids && ids->ne[1] == 1 && ids->ne[0] >= 1 && ids->ne[0] <= ec_n_cached();
    GGML_UNUSED(serve_max);
    ggml_ec_file_src fs;
    bool file_ok = src->name && src->name[0] && ggml_backend_ec_get_file_source(src->name, &fs);
    static int bn = 0;
    if (!ids_ok && bn < 3) {
        bn++;
        fprintf(stderr, "[ec-decline] %s ids_ne1=%lld serve_max=%lld elems=%lld C=%d\n", src->name,
                ids ? (long long) ids->ne[1] : -1, (long long) serve_max,
                ids ? (long long) ggml_nelements(ids) : -1, ec_n_cached());
    }
    if (!ids_ok || !file_ok) return 0;
    return ec_n_cached();
}

// [blocker B] widen the scheduler's MoE partial-copy entry test for Metal EC weights: accept
// node->src[0] == input (original host tensor) when the file-source registry knows this tensor.
// Belt: the node's ids must be this model's router ids (ne[1] bounded); the copy itself is looked
// up by the branch via tensor_id_copy, exactly as the stock path does.
static int hook_moe_allow_orig_src(const ggml_tensor * node, const ggml_tensor * input) {
    if (ec_n_cached() <= 0 || !node || !input || node->src[0] != input) return 0;
    return ec_node_serve_eligible(node);
}

// safety net: if the intercept cannot serve a call the copy phase engaged for, input_cpy was only
// partially filled — restore it with a full fill before the legacy GEMM reads.
static void hook_fallback_full(const ggml_tensor * dev, void * stream_v) {
    ggml_metal_ec_entry * e = ec_entry_lookup(dev);   // [blocker A] intercept passes the ORIGINAL host tensor
    GGML_ASSERT(e);
    GGML_ASSERT(e->host_base && e->dev_ptr);
    GGML_UNUSED(stream_v);   // Metal has no raw stream handle; the device queue is authoritative
    g_ec_fallbacks++;
    if (e->bounded || getenv("LLAMA_EC_DEBUG_SLOT_VERIFY")) {
        // a bounded buffer holds C slots, not n_expert; the served path is its only reader. A decline here means
        // the slot map was inconsistent - refilling fully is impossible, so refuse rather than read stale slabs.
        GGML_LOG_ERROR("%s: bounded entry %s cannot be refilled for a declined serve\n", __func__, ggml_get_name(e->host_ptr));
        GGML_ABORT("bounded arena decline");
    }

    id<MTLCommandQueue> dq = device_queue(e);
    {
        // drain first: nothing must still be in flight on the device queue before refilling
        id<MTLCommandBuffer> cb0 = [dq commandBuffer];
        [cb0 commit];
        [cb0 waitUntilCompleted];
    }

    if (e->disk && e->fd >= 0) {
        ec_fill_dev_slabs(e, e->dev_ptr, 0, (int32_t) e->n_expert - 1);
        return;   // completion-ordered on the same queue: legacy reads after these blits
    }
    if (!g_stage[0]) ec_stage_ring_ensure((id<MTLDevice>) ggml_metal_device_get_obj(e->mdev), e->slab);
    if (!g_stage[0]) return;   // no staging ring: caller must decline instead of refilling
    struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(ec_metal_ctx(e->dev_ptr->buffer, e->dev_ptr), e->dev_ptr);
    for (int32_t eid = 0; eid < e->n_expert; eid++) {
        memcpy(g_stage[0].contents, (const uint8_t *) e->host_base + (size_t) eid * e->slab, e->slab);
        id<MTLCommandBuffer> cb = [dq commandBuffer];
        id<MTLBlitCommandEncoder> enc = [cb blitCommandEncoder];
        [enc copyFromBuffer:g_stage[0] sourceOffset:0 toBuffer:(id<MTLBuffer>) bid_dst.metal
              destinationOffset:bid_dst.offs + (size_t) eid * e->slab size:e->slab];
        [enc endEncoding];
        [cb commit];
    }
    id<MTLCommandBuffer> cbf = [dq commandBuffer];
    [cbf commit];
    [cbf waitUntilCompleted];   // legacy path reads synchronously after this hook returns
}

static void ec_atexit_dump(void) {
    ec_profile_save();
    const uint64_t tot = g_ec_hits.load() + g_ec_misses.load();
    printf("[metal-expert-cache] C=%d engaged_lookups=%llu hits=%llu misses=%llu evicts=%llu hit_rate=%.4f fallbacks=%llu\n",
            ec_n_cached(), (unsigned long long) g_ec_calls.load(),
            (unsigned long long) g_ec_hits.load(), (unsigned long long) g_ec_misses.load(),
            (unsigned long long) g_ec_evicts.load(),
            tot ? (double) g_ec_hits.load() / (double) tot : 0.0,
            (unsigned long long) g_ec_fallbacks.load());
    printf("[metal-ec] verify_ok=%llu verify_bad=%llu host_stale=%llu audit_ok=%llu audit_bad=%llu\n",
            (unsigned long long) g_ec_verify_ok.load(), (unsigned long long) g_ec_verify_bad.load(),
            (unsigned long long) g_ec_hoststale.load(),
            (unsigned long long) g_ec_audit_ok.load(), (unsigned long long) g_ec_audit_bad.load());
    if (g_perf.on()) {
        auto ms = [](std::atomic<uint64_t> & a) { return (double) a.load() / 1e6; };
        printf("[ec-perf] pread %.0f ms/%llu | gate %.0f ms/%llu | blit %.0f ms/%llu | join %.0f ms/%llu | fillids %.0f ms/%llu | hit %.0f ms/%llu\\n",
               ms(g_perf.pread_ns), (unsigned long long) g_perf.pread_n.load(),
               ms(g_perf.gate_ns),  (unsigned long long) g_perf.gate_n.load(),
               ms(g_perf.blit_ns),  (unsigned long long) g_perf.blit_n.load(),
               ms(g_perf.join_ns),  (unsigned long long) g_perf.join_n.load(),
               ms(g_perf.fillids_ns),(unsigned long long) g_perf.fillids_n.load(),
               ms(g_perf.hit_ns),   (unsigned long long) g_perf.hit_n.load());
    }
    printf("[metal-ec] dbar: %llu ms / %llu waves (jobs %llu)\n",
           (unsigned long long) (g_dbar_ns / 1000000ULL), (unsigned long long) g_dbar_n.load(),
           (unsigned long long) g_dwave_n.load());
    printf("[metal-ec] pf: landed=%llu saved_as_hit=%llu dropped=%llu\n",
           (unsigned long long) g_ec_pf_fill.load(), (unsigned long long) g_ec_pf_hit.load(), (unsigned long long) g_ec_pf_drop.load());
    printf("[metal-ec] copy_bytes=%lluMB side_blits=%llu intercepts=%llu declines=%llu setups=%llu fillids=%llu\n",
            (unsigned long long) (g_ec_copy_bytes.load() >> 20),
            (unsigned long long) g_ec_side_copies.load(),
            (unsigned long long) g_ec_intercepts.load(),
            (unsigned long long) g_ec_declines.load(),
            (unsigned long long) g_ec_setups.load(),
            (unsigned long long) g_ec_fillids.load());
    fflush(stdout);
}

void ggml_metal_ec_count_intercept(void) { g_ec_intercepts++; }
void ggml_metal_ec_count_decline(void)   { g_ec_declines++; }
// device-registry registration: ONLY when LLAMA_EXPERT_CACHE > 0 (lazy probe — env can land after
void ggml_backend_ec_hooks_register_metal(void) {
    static bool done = false;
    if (done) return;
    if (ec_n_cached() <= 0) return;               // default OFF — ptr stays NULL, byte-identical path
    if (ggml_backend_ec_hooks_ptr) return;        // CUDA table wins in hybrid builds
    fprintf(stderr, "[ec-reg] hooks registered C=%d\n", ec_n_cached());
    done = true;
    {   // [policy] env: LLAMA_EC_PIN=0 disables the pin policy; LLAMA_EC_PROFILE=<path> persists freq
        const char * k = getenv("LLAMA_EC_PIN");
        g_ec_pin_k = k ? atoi(k) : (int) ec_n_cached() / 3;   // default: C/3 pins
        const char * pf = getenv("LLAMA_EC_PROFILE");
        if (pf && pf[0]) ec_profile_load(pf);
    }
    {   // [async] LLAMA_EC_ASYNC=0 keeps the synchronous demand path (regression/A-B control)
        const char * a = getenv("LLAMA_EC_ASYNC");
        g_async_on = !(a && atoi(a) == 0);
        static void * (*s_pf)(void *) = &ec_worker_thread;   // defined below; captured at reg
        (void) s_pf;
    }

    atexit(ec_atexit_dump);

    g_hooks.active            = hook_active;
    g_hooks.is_gpu_buffer     = hook_is_gpu_buffer;   // renamed hook (was is_cuda_buffer)
    g_hooks.disk              = hook_disk;
    g_hooks.copy_slabs        = hook_copy_slabs;
    g_hooks.bounded_ne2       = hook_bounded_ne2;
    g_hooks.force_copy        = hook_force_copy;
    g_hooks.copy_buft         = hook_copy_buft;
    g_hooks.assign_gpu        = hook_assign_gpu;
    g_hooks.setup             = hook_setup;
    g_hooks.lookup_or_evict   = hook_lookup_or_evict;
    g_hooks.copy_to_slot      = hook_copy_to_slot;
    g_hooks.fill_ids          = hook_fill_ids;
    g_hooks.swap_candidate    = nullptr;          // [M6-RR] deferred to v1.1 (scheduler guards NULL)
    g_hooks.rr_apply          = nullptr;          // [M6-RR] deferred to v1.1
    g_hooks.served_copy_phase = hook_served;
    g_hooks.engaged_this_call = hook_engaged;
    g_hooks.cache_slots       = hook_slots;
    g_hooks.arena             = hook_arena;
    g_hooks.ids_slot          = hook_ids_slot;
    g_hooks.fallback_full     = hook_fallback_full;
    g_hooks.gpu_begin         = nullptr;          // [M3-C diag] deferred to v1.1
    g_hooks.gpu_end           = nullptr;          // [M3-C diag] deferred to v1.1
    g_hooks.moe_allow_orig_src = hook_moe_allow_orig_src;   // [blocker B]

    ggml_backend_ec_hooks_ptr = &g_hooks;
}