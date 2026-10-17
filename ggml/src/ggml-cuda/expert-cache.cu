#include "common.cuh"
#include "expert-cache.cuh"

#include "ggml-backend-impl.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <cstdio>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <chrono>

#include "../ggml-ec-hooks.h"

static int g_ec_n = -1; // C slots per entry (layer); parsed LAZILY from LLAMA_EXPERT_CACHE on each
                        // query — model-params setenv() can land after CUDA device enumeration,
                        // so a one-shot read at init would miss it (learned the hard way).
static int ec_n_cached(void) {
    if (g_ec_n <= 0) { // re-probe until a positive value shows up (env may be set after init)
        const char * env = getenv("LLAMA_EXPERT_CACHE");
        g_ec_n = env ? atoi(env) : 0;
    }
    return g_ec_n;
}

// [M2-lite] hit/miss counters — the design's core unknown is temporal locality of routing.
static std::atomic<uint64_t> g_ec_hits{0}, g_ec_misses{0}, g_ec_evicts{0}, g_ec_calls{0};
static std::atomic<uint64_t> g_ec_fallbacks{0};
static std::atomic<uint64_t> g_ec_pinned_bytes{0};
static std::atomic<uint64_t> g_ec_copy_bytes{0};
static std::atomic<uint64_t> g_ec_copy_ns{0};
static std::atomic<uint64_t> g_t_setup{0}, g_t_lookev{0}, g_t_fillids{0};
static void ec_time(std::atomic<uint64_t> & to, const std::chrono::steady_clock::time_point & t0) {
    to += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();
}
static std::atomic<uint64_t> g_ec_intercepts{0};   // mul_mat_id GEMMs actually served from arena
static std::atomic<uint64_t> g_ec_declines{0};      // copy phase engaged but intercept declined
static std::atomic<uint64_t> g_ec_setups{0};        // copy-block entry count (ec_on decode calls)
static std::atomic<uint64_t> g_ec_fillids{0};       // fill_ids completions
static std::atomic<uint64_t> g_ec_gate_noserve{0};  // mul_mat_id CUDA calls where partial_fill==0
// [M3-C diag] per-layer served-GEMM drain time (LLAMA_EXPERT_CACHE_SYNC=1)
static std::atomic<uint64_t> g_layer_sync_ns{0}, g_layer_sync_cnt{0};
void ggml_cuda_ec_count_layer_sync(uint64_t ns) { g_layer_sync_ns += ns; g_layer_sync_cnt++; }
void ggml_cuda_ec_count_intercept(void) { g_ec_intercepts++; }
void ggml_cuda_ec_count_decline(void)  { g_ec_declines++; }
void ggml_cuda_ec_count_gate_noserve(void) { g_ec_gate_noserve++; }
static std::atomic<uint64_t> g_ec_graph_launches{0};  // [M3-D] cudaGraphLaunch under EC mode
void ggml_cuda_ec_count_graph_launch(void) { g_ec_graph_launches++; }
// [M3-C diag] GPU busy-time via cudaEvent brackets around split compute
// (LLAMA_EXPERT_CACHE_GPUTIME=1). Record begin/end on the backend stream at split launch;
// reap opportunistically from hook_setup with cudaEventQuery (never blocks). If
// gpu_busy_ns/token ~= wall/token -> GPU-bound; if << -> host-bound.
#define EC_GPU_RING 4096
static cudaEvent_t g_gtev[EC_GPU_RING], g_gtet[EC_GPU_RING];
static bool        g_gtpair_init[EC_GPU_RING] = {};
static uint64_t    g_gthead = 0, g_gtail = 0;
static std::atomic<uint64_t> g_gpu_busy_ns{0}, g_gpu_meas_splits{0}, g_gpu_dropped{0};
static std::atomic<uint64_t> g_gpu_pairs_begun{0};
static cudaStream_t stream_for(ggml_backend_t backend);  // fwd (defined below with the engine)

static cudaStream_t g_gt_stream[EC_GPU_RING] = {};  // stream captured at begin; end reuses it

// [M3-C diag] per-CUDA-API host timers: the split table says host time lives in "trivial"
// calls (setup=1.2ms?!). Time EVERY engine-side CUDA API to find the poison call.
static std::atomic<uint64_t> g_a_evr{0}, g_a_evr_n{0};       // cudaEventRecord
static std::atomic<uint64_t> g_a_evq{0}, g_a_evq_n{0};       // cudaEventQuery
static std::atomic<uint64_t> g_a_swe{0}, g_a_swe_n{0};       // cudaStreamWaitEvent
static std::atomic<uint64_t> g_a_cma{0}, g_a_cma_n{0};       // cudaMemcpyAsync
static std::atomic<uint64_t> g_a_cmb{0}, g_a_cmb_n{0};       // cudaMemsetAsync
static std::atomic<uint64_t> g_a_evtc{0}, g_a_evtc_n{0};     // cudaEventCreateWithFlags
static std::atomic<uint64_t> g_a_stc{0},  g_a_stc_n{0};      // cudaStreamCreate
static std::atomic<uint64_t> g_a_ghs{0},  g_a_ghs_n{0};      // cudaGetLastError (sticky sweep)
#define EC_API(call, nsvar, nvar) do {                                        \
    const auto _t0 = std::chrono::steady_clock::now();                         \
    call;                                                                      \
    nsvar += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(  \
        std::chrono::steady_clock::now() - _t0).count(); nvar++;               \
} while (0)

static void hook_gpu_begin(ggml_backend_t backend) {
    if (!ggml_backend_is_cuda(backend)) return;  // CPU splits: stream_for would mis-cast context
    const uint64_t idx = g_gthead % EC_GPU_RING;
    if (g_gtpair_init[idx]) {  // ring wrapped with unreaped pairs: drop, keep correctness
        g_gpu_dropped++;
        cudaEventDestroy(g_gtev[idx]); cudaEventDestroy(g_gtet[idx]);
        g_gtev[idx] = g_gtet[idx] = nullptr;
        g_gtpair_init[idx] = false;
    }
    cudaStream_t s = stream_for(backend);
    if (!g_gtev[idx])  { if (cudaEventCreateWithFlags(&g_gtev[idx], cudaEventDisableTiming) != cudaSuccess) { cudaGetLastError(); return; } }
    if (!g_gtet[idx])  { if (cudaEventCreateWithFlags(&g_gtet[idx], cudaEventDisableTiming) != cudaSuccess) { cudaGetLastError(); return; } }
    if (cudaEventRecord(g_gtev[idx], s) != cudaSuccess) { cudaGetLastError(); return; }
    g_gt_stream[idx] = s;   // end() records on THIS stream, not the (possibly different) backend arg's
    g_gtpair_init[idx] = true;
    g_gthead++;
    g_gpu_pairs_begun++;
}
static void hook_gpu_end(ggml_backend_t backend) {
    if (!ggml_backend_is_cuda(backend)) return;  // must mirror begin (same split, same guard)
    if (g_gthead == 0) return;
    const uint64_t idx = (g_gthead - 1) % EC_GPU_RING;
    if (!g_gtpair_init[idx]) return;
    if (cudaEventRecord(g_gtet[idx], g_gt_stream[idx]) != cudaSuccess) cudaGetLastError();
}
static void ec_gpu_reap(void) {
    while (g_gtail < g_gthead) {
        const uint64_t tidx = g_gtail % EC_GPU_RING;
        if (!g_gtpair_init[tidx]) { g_gtail++; continue; }
        if (cudaEventQuery(g_gtet[tidx]) != cudaSuccess) break;
        float ms = 0.f;
        if (cudaEventElapsedTime(&ms, g_gtev[tidx], g_gtet[tidx]) == cudaSuccess) {
            g_gpu_busy_ns += (uint64_t) (ms * 1e6);
            g_gpu_meas_splits++;
        }
        g_gtpair_init[tidx] = false;
        g_gtail++;
    }
}

static void ec_dump_names(void);  // fwd — needs g_entries, defined below

static void ec_atexit_dump(void) {
    // [M3-A] sync-site table (ggml-backend.cpp): where the host burns wall time per token
    static const char * site_name[GGML_EC_SITE_COUNT] = {
        "split_sync", "input_sync", "ids_sync", "ev_wait", "ev_sync", "graph_compute",
        "graph_drain", "layer_sync", "split_wall", "sched_sync", "gc_gpu", "gc_cpu",
        "pre_build", "split_pre", "split_post", "hook_setup", "hook_fill", "hook_precopy",
        "copy_fallback", "legacy_set", "inputs_total", "alloc_graph", "sched_outer",
        "in_prep", "moe_branch", "reg_copy", "flag_input", "tensor_copy", "predicate" };
    printf("[ec-syncprof] site                 calls      ms      avg_us\n");
    for (int s = 0; s < GGML_EC_SITE_COUNT; s++) {
        const uint64_t ns = g_ec_prof_ns[s].load();
        const uint64_t cn = g_ec_prof_cnt[s].load();
        printf("[ec-syncprof] %-18s %8llu %8.1f %8.1f\n", site_name[s],
                (unsigned long long) cn, ns / 1e6, cn ? ns / 1e3 / (double) cn : 0.0);
    }
    if (g_layer_sync_cnt.load()) {
        printf("[ec-syncprof] served_layer_sync  %8llu %8.1f %8.1f  (LLAMA_EXPERT_CACHE_SYNC)\n",
                (unsigned long long) g_layer_sync_cnt.load(), g_layer_sync_ns.load() / 1e6,
                g_layer_sync_ns.load() / 1e3 / (double) g_layer_sync_cnt.load());
    }
    if (g_gpu_pairs_begun.load()) {
        printf("[ec-syncprof] gpu_busy total %.1f ms over %llu/%llu splits reaped (dropped %llu)\n",
                g_gpu_busy_ns.load() / 1e6, (unsigned long long) g_gpu_meas_splits.load(),
                (unsigned long long) g_gpu_pairs_begun.load(),
                (unsigned long long) g_gpu_dropped.load());
    }
    const uint64_t tot = g_ec_hits.load() + g_ec_misses.load();
    printf("[expert-cache] C=%d engaged_lookups=%llu hits=%llu misses=%llu evicts=%llu hit_rate=%.4f fallbacks=%llu\n",
            ec_n_cached(), (unsigned long long) g_ec_calls.load(),
            (unsigned long long) g_ec_hits.load(), (unsigned long long) g_ec_misses.load(),
            (unsigned long long) g_ec_evicts.load(),
            tot ? (double) g_ec_hits.load() / (double) tot : 0.0,
            (unsigned long long) g_ec_fallbacks.load());
    printf("[expert-cache] pinned=%lluMB miss_copy_issued=%lluMB issue_host_time_ms=%llu\n",
            (unsigned long long) (g_ec_pinned_bytes.load() >> 20),
            (unsigned long long) (g_ec_copy_bytes.load() >> 20),
            (unsigned long long) (g_ec_copy_ns.load() / 1000000));
    printf("[expert-cache] host_ms: setup=%.1f lookev=%.1f fillids=%.1f (copy see below)\n",
            g_t_setup.load() / 1e6, g_t_lookev.load() / 1e6, g_t_fillids.load() / 1e6);
    ec_dump_names();  // [M3-G diag] defined after g_entries; no-op unless LLAMA_EC_DUMP_NAMES
    printf("[ec-api] evq=%llu/%llu reap=%llu/%llu cma=%llu/%llu swe=%llu/%llu evrc=%llu/%llu\n",
            (unsigned long long) g_a_evq.load(), (unsigned long long) g_a_evq_n.load(),
            (unsigned long long) g_a_cmb.load(), (unsigned long long) g_a_cmb_n.load(),
            (unsigned long long) g_a_cma.load(), (unsigned long long) g_a_cma_n.load(),
            (unsigned long long) g_a_swe.load(), (unsigned long long) g_a_swe_n.load(),
            (unsigned long long) g_a_evr.load(), (unsigned long long) g_a_evr_n.load());
    printf("[expert-cache] intercepts=%llu declines=%llu setups=%llu fillids=%llu gate_noserve=%llu graph_launches=%llu\n",
            (unsigned long long) g_ec_intercepts.load(), (unsigned long long) g_ec_declines.load(),
            (unsigned long long) g_ec_setups.load(), (unsigned long long) g_ec_fillids.load(),
            (unsigned long long) g_ec_gate_noserve.load(), (unsigned long long) g_ec_graph_launches.load());
}

static ggml_backend_ec_hooks g_hooks;
// note: entries intentionally live for process lifetime (freed with the process)

int ggml_cuda_ec_active(void) {
    return ec_n_cached() > 0;
}

// entries keyed by the HOST expert tensor pointer (stable across calls; input_cpy pointers rotate
// through sched copy slots). dev_idx maps the current device-side copy back to its entry.
static std::unordered_map<const ggml_tensor *, ggml_cuda_ec_entry> g_entries;

// [M3-G diag] LLAMA_EC_DUMP_NAMES=<substr>: per-entry routing stats for MoE tensors whose name
// contains <substr> (e.g. "nextn" / "ffn" / "blk.52") — answers "is the MTP draft head's MoE
// registered & routed at all?". Runs from ec_atexit_dump.
static void ec_dump_names(void) {
    const char * dn = getenv("LLAMA_EC_DUMP_NAMES");
    if (!dn) return;
    for (auto & kv : g_entries) {
        const ggml_cuda_ec_entry & e = kv.second;
        if (!e.host_ptr || !e.host_ptr->name) continue;
        if (!strstr(e.host_ptr->name, dn)) continue;
        unsigned long long fsum = 0, fmax = 0;
        int resident = 0;
        for (uint64_t f : e.freq) { fsum += f; if (f > fmax) fmax = f; }
        for (int64_t s : e.slot_of_id) if (s >= 0) resident++;
        printf("[ec-name] %-44s n_expert=%lld slab=%zu engaged=%d resident=%d route_counted=%llu hottest=%llu\n",
                e.host_ptr->name, (long long) e.n_expert, e.slab, e.engaged, resident,
                fsum, fmax);
    }
    fflush(stdout);
}
static std::unordered_map<const ggml_tensor *, ggml_cuda_ec_entry *> g_dev_idx;

static int hook_is_cuda_buffer(const ggml_tensor * t) {
    // [M3-F perf] hot path (4x per MoE input x 48 layers x token): the buft->device walk is
    // pure-registry but repeated identically for the same buffer every call. Memoize on buffer
    // ptr (buffers outlive the sched for the model's life). Strata-style: ask once, cache plan.
    // std::hash<const T*> is DELETED in libstdc++ — key on the non-const pointer, cast once.
    static std::unordered_map<ggml_backend_buffer_t, char> memo;  // 0/1 cached verdict
    const ggml_backend_buffer_t buf = t->buffer;
    if (!buf) return 0;
    auto it = memo.find(buf);
    if (it != memo.end()) return it->second;
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buf));
    const char r = (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) ? 1 : 0;
    memo.emplace(buf, r);
    return r;
}

ggml_cuda_ec_entry * ggml_cuda_ec_find(const ggml_tensor * dev) {
    auto it = g_dev_idx.find(dev);
    return it == g_dev_idx.end() ? nullptr : it->second;
}

static ggml_cuda_ec_entry * lookup_or_create(const ggml_tensor * host, const ggml_tensor * dev) {
    auto it = g_entries.find(host);
    if (it != g_entries.end()) {
        ggml_cuda_ec_entry * e = &it->second;
        // this call's device-side copy is (possibly) a different sched slot — re-index it
        e->dev_ptr = dev;
        g_dev_idx[dev] = e;
        return e;
    }
    GGML_ASSERT(host->buffer && !hook_is_cuda_buffer(host)); // entry mirrors HOST weights, not device

    ggml_cuda_ec_entry e;
    e.host_ptr   = host;
    e.dev_ptr    = dev;
    e.host_base  = host->data; // cudaHostAlloc'd pinned base
    e.slab       = host->nb[2];
    e.n_expert   = host->ne[2];
    e.C          = ec_n_cached();
    // [M3-G] full-residency override for chosen layers, e.g. LLAMA_EC_PIN_LAYERS=blk.52 pins the
    // whole MTP draft-head MoE (Ferrol: "I bet itd do well if it was [in VRAM]"). Arena grows to
    // n_expert * slab — check VRAM headroom (nextn up+down ~= 0.9 GB, well inside budget).
    if (host->name) {
        if (const char * pl = getenv("LLAMA_EC_PIN_LAYERS")) {
            // comma/semicolon list of name prefixes; pinned layers bypass eviction entirely
            size_t i = 0;
            while (i < strlen(pl)) {
                size_t j = i;
                while (j < strlen(pl) && pl[j] != ',' && pl[j] != ';') j++;
                size_t len = j - i;
                if (len && strncmp(pl + i, host->name, len) == 0) { e.C = e.n_expert; break; }
                i = j + 1;
            }
        }
    }

    CUDA_CHECK(cudaMalloc((void **) &e.arena, (size_t) e.C * e.slab));
    // zero once so block-quant padding inside a slab is never NaN (MMQ/MMVQ safety)
    CUDA_CHECK(cudaMemset(e.arena, 0, (size_t) e.C * e.slab));

    e.slots.assign(e.C, ggml_cuda_ec_slot{});
    e.slot_of_id.assign(e.n_expert, -1);
    e.freq.assign(e.n_expert, 0);
    {   // [M3-E] when to prestage (decode calls per layer) and how much (MB of expert bytes)
        const char * genv = getenv("LLAMA_EC_PREHEAT_GEN");
        e.preheat_gen = genv ? strtoull(genv, nullptr, 10) : 48;
    }

    // One-time pinned mirror of the expert slab array.
    // [M3-F ROOT CAUSE, Oct 16] This used to run HERE — on the FIRST ENCODED CALL for the layer,
    // i.e. inside decode: cudaHostAlloc(462MB per entry; 20.9GB total) + a synchronous 20.9GB
    // memcpy STALLED the host thread for ~14 s total (ec-api: lookup_or_create=14164ms/11822).
    // That inflated every EC measurement (~14s of split-loop wall inside the token window),
    // hid the true steady-state cost, and idled the GPU during bench — the "low utilization".
    // NO_COPY already proved copies aren't the wall, so the mirror is OFF by default now
    // (miss copies read the GGUF mmap pages directly, ~13.7 GB/s pageable). Opt in: LLAMA_EC_PINNED_MIRROR=1.
    if (getenv("LLAMA_EC_PINNED_MIRROR")) {
        const size_t region = (size_t) e.n_expert * e.slab;
        char * mirror = nullptr;
        cudaError_t err = cudaHostAlloc((void **) &mirror, region, cudaHostAllocDefault);
        if (err == cudaSuccess) {
            // parallel bulk copy: this is model-load time, do it with plain memcpy threads later if slow
            memcpy(mirror, e.host_base, region);
            e.pinned = mirror;
            g_ec_pinned_bytes += region;
        } else {
            static int reported = 0;
            if (!reported++) {
                fprintf(stderr, "[expert-cache] cudaHostAlloc FAILED (%s) len=%.2fGB -> falling back to pageable source\n",
                        cudaGetErrorString(err), (double) region / 1e9);
            }
            cudaGetLastError();
            e.pinned = nullptr; // hook_copy_to_slot falls back to host_base (slow but correct)
        }
    }

    // [M3-B] side copy stream + fork event. Miss fills run HERE, off the kernel stream: the
    // scheduler's per-layer event-wait drains the kernel stream, and while copies lived on it
    // every one of those waits paid full PCIe transfer time (measured ev_wait 317us x 48/token).
    // Fork ordering: a copy never overwrites a slot an enqueued GEMM may still be reading,
    // because the copy waits on an event recorded on the backend stream AT ISSUE TIME.
    CUDA_CHECK(cudaStreamCreateWithFlags(&e.copy_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&e.fork_event, cudaEventDisableTiming));

    auto res = g_entries.emplace(host, std::move(e));
    g_dev_idx[dev] = &res.first->second;
    return &res.first->second;
}

static cudaStream_t stream_for(ggml_backend_t backend) {
    ggml_backend_cuda_context * cctx = (ggml_backend_cuda_context *) backend->context;
    return cctx->stream();
}

// --- hooks called from the ggml-backend.cpp sched copy phase ---

static int hook_active(void) {
    return ec_n_cached() > 0;
}


// (LLAMA_EC_SPIN resident-kernel keep-warm probe removed 2026-10-16: deadlocked the first
//  stream sync during its trial; P-state theory was disproven by dmon — stock 33 t/s also
//  runs at P8/210MHz. Verdict in NOW.md Oct 16 addendum.)

static int hook_setup(ggml_backend_t backend, const ggml_tensor * host, ggml_tensor * dev, int64_t n_tokens, int64_t n_ids_elems) {
    const auto t0 = std::chrono::steady_clock::now();
    struct T { std::atomic<uint64_t> * to; const std::chrono::steady_clock::time_point t0;
               ~T() { ec_time(*to, t0); } } tr { &g_t_setup, t0 };
    g_ec_setups++;
    const auto tr0 = std::chrono::steady_clock::now();
    ec_gpu_reap();
    g_a_evq += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - tr0).count(); g_a_evq_n++;  // reap timer
    const auto tl0 = std::chrono::steady_clock::now();
    ggml_cuda_ec_entry * e = lookup_or_create(host, dev);
    g_a_cmb += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - tl0).count(); g_a_cmb_n++;  // lookup_or_create timer
    e->gen++;
    e->ids_expect = n_ids_elems;
    e->partial_fill = 0;
    // engage ONLY for decode-sized ubatches (mirrors the intercept's ne2 <= MMVQ_MAX_BATCH gate).
    // Prefill keeps the legacy input_cpy copy path byte-identically.
    // [M3-G FIX] decline when the call's routed width exceeds the arena: each unique routed id
    // stamps a slot and stamps are never evictable mid-call, so C < n_ids_elems can leave zero
    // victims (observed: GGML_ASSERT(victim >= 0) abort at C=16 with SERVE_MAX=4 => 24 stamps).
    // Declining is safe and exact: engaged=0 -> no serve, legacy full-slab path (stock behavior).
    e->engaged   = (n_tokens <= 8 && e->C >= n_ids_elems) ? 1 : 0;
    e->last_ids  = nullptr;
    e->last_ids_n = 0;
    (void) backend;
    return 1;
}

static int hook_lookup_or_evict(const ggml_tensor * dev, int32_t id, int64_t * out_slot, int * out_need_copy) {
    const auto t0 = std::chrono::steady_clock::now();
    struct T { std::atomic<uint64_t> * to; const std::chrono::steady_clock::time_point t0;
               ~T() { ec_time(*to, t0); } } tr { &g_t_lookev, t0 };
    ggml_cuda_ec_entry * e = ggml_cuda_ec_find(dev);
    GGML_ASSERT(e && id >= 0 && id < e->n_expert);

    if (e->slot_of_id[id] >= 0) {
        int64_t s = e->slot_of_id[id];
        e->slots[s].stamp_gen = e->gen;      // touched this call: never evict
        e->slots[s].last_use  = e->gen;
        *out_slot      = s;
        *out_need_copy = 0;
        e->partial_fill = 1;                 // legacy full copy was skipped for this slab
        g_ec_hits++;
        g_ec_calls++;
        return 1;
    }

    // victim: empty slot > LRU-oldest unstamped > LRU-oldest (never stamp this call's own ids).
    // [M3-E] hot (pre-staged) slots are evicted only after every cold candidate is exhausted.
    // In-flight copies are safe to overwrite (same-stream ordering), but prefer finished ones.
    int64_t victim = -1;
    for (int64_t s = 0; s < e->C; s++) {
        if (e->slots[s].expert_id < 0) { victim = s; break; }
    }
    if (victim < 0) {
        for (int pass = 0; pass < 3 && victim < 0; pass++) {
            uint64_t oldest = UINT64_MAX;
            for (int64_t s = 0; s < e->C; s++) {
                const ggml_cuda_ec_slot & sl = e->slots[s];
                if (sl.stamp_gen == e->gen) continue; // in use by this call
                if (sl.hot && pass == 0) continue;    // keep hot set while any cold victim exists
                const bool inflight = sl.in_flight && cudaEventQuery(sl.in_flight) == cudaErrorNotReady;
                if (inflight && pass < 2 && oldest != UINT64_MAX) continue; // prefer settled slots
                if (sl.last_use < oldest) { oldest = sl.last_use; victim = s; }
            }
        }
    }
    // pass 3 (above) allows hot + in-flight victims: same-stream ordering makes overwriting an
    // in-flight copy safe as long as the slot is not being READ this call (stamp excludes those).
    GGML_ASSERT(victim >= 0); // only possible if EVERY slot is stamp==gen: C < routed width — impossible at C>=8

    ggml_cuda_ec_slot & sl = e->slots[victim];
    if (sl.expert_id >= 0) {
        e->slot_of_id[sl.expert_id] = -1;
        g_ec_evicts++;
    }
    sl.expert_id   = id;
    sl.hot         = false;   // fresh occupants are cold; only preheat marks hot
    sl.stamp_gen   = e->gen;
    sl.last_use    = e->gen;
    e->slot_of_id[id] = victim;
    *out_slot      = victim;
    *out_need_copy = 1;
    e->partial_fill = 1;                     // legacy full copy was skipped for this slab
    g_ec_misses++;
    g_ec_calls++;
    return 1;
}

static int hook_copy_to_slot(ggml_backend_t backend, const ggml_tensor * dev, int32_t id, int64_t slot) {
    ggml_cuda_ec_entry * e = ggml_cuda_ec_find(dev);
    GGML_ASSERT(e && e->host_base && slot >= 0 && slot < e->C);
    cudaStream_t kstream = stream_for(backend);
    const uint8_t * src = (const uint8_t *) (e->pinned ? e->pinned : e->host_base) + (size_t) id * e->slab;
    uint8_t * dst = e->arena + (size_t) slot * e->slab;
    const auto t0 = std::chrono::steady_clock::now();
    // A/B knob: LLAMA_EC_COPY_SAME=1 restores the M2 same-stream copy (stream ordering makes
    // the join free) to isolate what the M3-B side-stream fork/join costs.
    static const bool same_stream = getenv("LLAMA_EC_COPY_SAME") != nullptr;
    // TIMING-ONLY ROOFLINE PROBE: LLAMA_EC_NO_COPY=1 makes miss copies no-ops (stale slot data,
    // output is GARBAGE — never ship, never trust text). If t/s jumps, PCIe copies are the wall;
    // if it doesn't move, host/scheduler overhead is. Join-waits on never-recorded events
    // return "completed" immediately, so the probe is self-consistent.
    static const bool no_copy = getenv("LLAMA_EC_NO_COPY") != nullptr;
    ggml_cuda_ec_slot & sl = e->slots[slot];
    if (no_copy) {
        sl.loading = false;
        g_ec_copy_bytes += 0; // deliberately not counted
        return 1;
    }
    // [M3-F] Without the pinned mirror the copy SOURCE is GGUF mmap pages: cudaMemcpyAsync from
    // pageable/file-backed memory is host-blocking (~13.7 GB/s, 512MB of it per token) — putting
    // it on a side stream buys nothing (measured wash) and its fork/join events add driver traffic.
    // Only mirror-backed (true async pinned DMA) copies belong on the side stream.
    const bool use_side = !same_stream && e->pinned != nullptr;
    if (!use_side) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, e->slab, cudaMemcpyHostToDevice, kstream));
        if (sl.in_flight) { cudaEventRecord(sl.in_flight, kstream); }
        sl.loading = false;
    } else {
        // [M3-B] fork: the slot may still be read by GEMMs enqueued on the kernel stream (any call
        // that routed to the previous occupant). Wait on an event recorded NOW on that stream before
        // overwriting. join (fill_ids) makes the kernel stream wait on the copy before the next read.
        CUDA_CHECK(cudaEventRecord(e->fork_event, kstream));
        CUDA_CHECK(cudaStreamWaitEvent(e->copy_stream, e->fork_event, 0));
        CUDA_CHECK(cudaMemcpyAsync(dst, src, e->slab, cudaMemcpyHostToDevice, e->copy_stream));
        if (!sl.in_flight) {
            CUDA_CHECK(cudaEventCreateWithFlags(&sl.in_flight, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(sl.in_flight, e->copy_stream));
        sl.loading = true; // cleared when fill_ids joins the kernel stream onto this slot
        e->side_copies++;
        e->side_bytes += e->slab;
    }
    g_ec_copy_bytes += e->slab;
    g_ec_copy_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();
    return 1;
}

static void ec_preheat(ggml_backend_t backend, ggml_cuda_ec_entry * e) {
    // [M3-E] one-time (per layer) hot-set prestage (Ferrol's M3-E directive): rank experts by
    // observed routing frequency, bulk-copy the hottest K into the arena on copy_stream, mark
    // them hot so LRU drains cold slots first. Budget = LLAMA_EC_PRELOAD_MB total MB / entries.
    e->preheated = true;
    static const long long preload_mb = [] {
        const char * s = getenv("LLAMA_EC_PRELOAD_MB");
        return s ? strtoll(s, nullptr, 10) : 4096;   // Ferrol: "about 4 gb"
    }();
    if (preload_mb <= 0 || e->slab == 0) return;
    const size_t n_entries = std::max<size_t>(1, g_entries.size());
    const int64_t k = std::min<int64_t>(
        e->n_expert, (int64_t) ((double) preload_mb * 1e6 / (double) n_entries / (double) e->slab));
    if (k <= 0) return;

    std::vector<int32_t> order(e->n_expert);
    for (int32_t i = 0; i < e->n_expert; i++) order[i] = i;
    const std::vector<uint64_t> & f = e->freq;
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return f[a] > f[b]; });

    for (int64_t i = 0; i < k; i++) {
        const int32_t id = order[i];
        int64_t slot = -1; int need_copy = 0;
        // reuse the normal reservation path (victim rules incl. hot-protection), then promote
        if (!hook_lookup_or_evict(e->dev_ptr, id, &slot, &need_copy)) continue;
        if (need_copy) {
            hook_copy_to_slot(backend, e->dev_ptr, id, slot);   // side-stream + event gate
            e->preheat_copied++;
        }
        e->slots[slot].hot = true;
        e->slots[slot].stamp_gen = 0; // preheated slots aren't "in use this call"; LRU by last_use
    }
    fprintf(stderr, "[expert-cache] preheat: %lld hot experts x %.1fMB copied=%llu (layer gen=%llu)\n",
            (long long) k, e->slab / 1e6, (unsigned long long) e->preheat_copied,
            (unsigned long long) e->gen);
}

static int hook_fill_ids(ggml_backend_t backend, const ggml_tensor * dev, const int32_t * host_ids, int n) {
    const auto t0 = std::chrono::steady_clock::now();
    struct T { std::atomic<uint64_t> * to; const std::chrono::steady_clock::time_point t0;
               ~T() { ec_time(*to, t0); } } tr { &g_t_fillids, t0 };
    ggml_cuda_ec_entry * e = ggml_cuda_ec_find(dev);
    GGML_ASSERT(e && e->engaged);
    GGML_ASSERT(n > 0);

    if (e->ids_ring_cap_ints < n) { // grow all ring slots (tiny buffers; decode only)
        for (int r = 0; r < GGML_CUDA_EC_ID_RING; r++) {
            if (e->ids_ring[r]) { cudaFree(e->ids_ring[r]); }
        }
        e->ids_ring_cap_ints = std::max<int64_t>(n, 512);
        for (int r = 0; r < GGML_CUDA_EC_ID_RING; r++) {
            CUDA_CHECK(cudaMalloc((void **) &e->ids_ring[r], e->ids_ring_cap_ints * sizeof(int32_t)));
        }
    }

    // [M3-D] graphs mode: the captured mmvq bakes the ring buffer POINTER from capture time.
    // Rotate the ring and replay reads STALE ids (Strata's KL-10x trap). Pin one slot instead:
    // single in-flight decode call + stream-ordered upload = same-buffer reuse is safe.
    static const bool ec_graphs = getenv("LLAMA_EC_GRAPHS") != nullptr;
    const int slot_idx = ec_graphs ? 0 : e->ids_ring_next;
    if (!ec_graphs) e->ids_ring_next = (e->ids_ring_next + 1) % GGML_CUDA_EC_ID_RING;
    int32_t * dst_dev = e->ids_ring[slot_idx];

    // stage + upload this call's slot map. [M3-F ROOT CAUSE] stage_pinned was REUSED across
    // calls while its pinned-DMA upload was still in flight: the next fill_ids overwrote it
    // before the GPU read it (host runs ahead at decode speed) -> ring served LAST call's
    // slots, mid-decode, nondeterministically. Every sync-probe masked it. Eager path now
    // uploads from a thread-local PAGEABLE buffer: cudaMemcpyAsync from pageable memory
    // host-stages before returning, so immediate reuse is safe. (Buffer is tiny: <=8x6 ints.)
    // Graph capture keeps the pinned stage (pageable sources illegal under capture) — graphs
    // mode is host-paced by the logits readback, so the one-in-flight window is enforced.
    // AUTHORITATIVE resolution: every id routed by THIS call is resolved here (the sched copy
    // loop is only a prefetch — its bitset can lag when upstream reuses an ids tensor).
    // Resolution here issues the miss copy on the same stream, ordered ahead of its GEMM.
    cudaStreamCaptureStatus cap_stage = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream_for(backend), &cap_stage);
    thread_local std::vector<int32_t> stage_pageable;
    int32_t * stage;
    if (cap_stage != cudaStreamCaptureStatusNone) {
        if (e->stage_cap_ints < n) {
            if (e->stage_pinned) cudaFreeHost(e->stage_pinned);
            e->stage_cap_ints = std::max<int64_t>(n, 512);
            CUDA_CHECK(cudaHostAlloc((void **) &e->stage_pinned,
                                     e->stage_cap_ints * sizeof(int32_t), cudaHostAllocDefault));
        }
        stage = e->stage_pinned;
    } else {
        if (stage_pageable.size() < (size_t) n) stage_pageable.resize(std::max<int64_t>(n, 512));
        stage = stage_pageable.data();
    }
    // [M3-E] frequency stats for the hot set (accumulated from call 1 = the sampling window)
    // + one-shot preheat trigger at warmup gen.
    if (!e->preheated && e->gen >= e->preheat_gen) ec_preheat(backend, e);
    for (int i = 0; i < n; i++) {
        const int32_t id = host_ids[i];
        if (id >= 0 && id < e->n_expert) e->freq[id]++;
        if (id < 0 || id >= e->n_expert) {
            // scheduler fed routing outside [0, n_expert) — decline the cache path for this call;
            // the fallback_full + legacy GEMM keeps output correct. Log once for forensics.
            static bool warned = false;
            if (!warned) {
                warned = true;
                fprintf(stderr, "[expert-cache] BAD id=%d at i=%d/%d (n_expert=%lld, gen=%llu) tensor=%s\n",
                        id, i, n, (long long) e->n_expert, (unsigned long long) e->gen,
                        ggml_get_name(e->host_ptr));
            }
            e->engaged    = 0;
            e->last_ids   = nullptr;
            e->last_ids_n = 0;
            return 0;
        }
        int64_t s = e->slot_of_id[id];
        if (s < 0) {
            int need_copy = 0;
            hook_lookup_or_evict(dev, id, &s, &need_copy);
            if (need_copy) {
                hook_copy_to_slot(backend, dev, id, s);
            }
        }
        stage[i] = (int32_t) s;
    }
    // [M3-B] join: the kernel stream must not read any slot whose side-stream copy is still
    // loading — including slots that HIT this call but were mid-fill from an earlier miss.
    // Event-waits are GPU-side; the host never blocks. Dedupe: one wait per repeated slot.
    for (int i = 0; i < n; i++) {
        const int32_t s = stage[i];
        ggml_cuda_ec_slot & sl = e->slots[s];
        if (!sl.loading || !sl.in_flight) continue;
        bool dup = false;
        for (int j = 0; j < i && !dup; j++) dup = (stage[j] == s);
        if (dup) continue;
        CUDA_CHECK(cudaStreamWaitEvent(stream_for(backend), sl.in_flight, 0));
        sl.loading = false;
    }
    // ring reuse is stream-ordered behind any GEMM still reading the older slot
    const cudaError_t up_err = cudaMemcpyAsync(dst_dev, stage, (size_t) n * sizeof(int32_t),
                                               cudaMemcpyHostToDevice, stream_for(backend));
    static const bool ec_verify = getenv("LLAMA_EC_VERIFY") != nullptr;
    if (ec_verify && e->gen <= 24) {
        // sync EVERY stream that could still be writing (backend stream ordered the ring upload;
        // side-stream miss copies) BEFORE reading back — else the probe itself races the DMA.
        static std::vector<uint8_t> a_buf, b_buf;
        if (a_buf.size() < (size_t) e->slab) { a_buf.resize((size_t) e->slab); b_buf.resize((size_t) e->slab); }
        cudaStreamSynchronize(stream_for(backend));
        if (e->copy_stream) cudaStreamSynchronize(e->copy_stream);
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream_for(backend), &cap);
        const cudaError_t rd_err = (cudaError_t) cudaMemcpy(a_buf.data(), dst_dev, n * sizeof(int32_t), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[ec-verify] UP %s gen=%llu n=%d up=%s rd=%s cap=%d dst=%p stage=%p cap_ints=%lld\n",
                ggml_get_name(e->host_ptr), (unsigned long long) e->gen, n,
                cudaGetErrorString(up_err), cudaGetErrorString(rd_err), (int) cap,
                (void *) dst_dev, (void *) stage, (long long) e->ids_ring_cap_ints);
        fprintf(stderr, "[ec-verify] RING %s ring=[%d,%d,%d,%d,%d,%d] stage=[%d,%d,%d,%d,%d,%d]\n",
                ggml_get_name(e->host_ptr),
                n > 0 ? ((const int32_t *) a_buf.data())[0] : -1, n > 1 ? ((const int32_t *) a_buf.data())[1] : -1,
                n > 2 ? ((const int32_t *) a_buf.data())[2] : -1, n > 3 ? ((const int32_t *) a_buf.data())[3] : -1,
                n > 4 ? ((const int32_t *) a_buf.data())[4] : -1, n > 5 ? ((const int32_t *) a_buf.data())[5] : -1,
                n > 0 ? stage[0] : -1, n > 1 ? stage[1] : -1, n > 2 ? stage[2] : -1,
                n > 3 ? stage[3] : -1, n > 4 ? stage[4] : -1, n > 5 ? stage[5] : -1);
        for (int i = 0; i < n; i++) {
            const int64_t s = stage[i]; const int32_t id = host_ids[i];
            cudaMemcpy(a_buf.data(), e->arena + (size_t) s * e->slab, e->slab, cudaMemcpyDeviceToHost);
            memcpy(b_buf.data(), (const uint8_t *) e->host_base + (size_t) id * e->slab, e->slab);
            if (memcmp(a_buf.data(), b_buf.data(), e->slab) != 0) {
                size_t off = 0; while (off < (size_t) e->slab && a_buf[off] == b_buf[off]) off++;
                fprintf(stderr, "[ec-verify] SLOT-DATA %s gen=%llu i=%d id=%d slot=%lld first_diff=%zu\n",
                        ggml_get_name(e->host_ptr), (unsigned long long) e->gen, i, id,
                        (long long) s, off);
            }
        }
    }
    e->last_ids   = dst_dev;
    e->last_ids_n = n;
    g_ec_fillids++;
    return 1;
}

// --- hooks called from the ggml-cuda.cu mul_mat_id intercept ---

static int hook_served(const ggml_tensor * dev) {
    // true when the copy phase touched input_cpy for this call: if we then fail to intercept,
    // the legacy GEMM would read a partially-filled buffer -> fallback_full refill is mandatory
    const ggml_cuda_ec_entry * e = ggml_cuda_ec_find(dev);
    return (e && e->partial_fill) ? 1 : 0;
}

static int hook_engaged(const ggml_tensor * dev) {
    const ggml_cuda_ec_entry * e = ggml_cuda_ec_find(dev);
    if (!e || !e->engaged || !e->last_ids) {
        static int logged = 0;
        if (logged < 3) {
            logged++;
            fprintf(stderr, "[ec-engaged-fail] e=%p engaged=%d last_ids=%p expect=%lld last_n=%lld\n",
                    (void *) e, e ? e->engaged : -1, e ? e->last_ids : nullptr,
                    e ? (long long) e->ids_expect : -1, e ? (long long) e->last_ids_n : -1);
        }
        return 0;
    }
    // the upload must match the ids tensor the kernel is about to read; any mismatch -> decline
    // the cache path (stay on legacy = correct, just slower)
    return (e->ids_expect == 0 || e->last_ids_n == e->ids_expect) ? 1 : 0;
}

static int64_t hook_slots(const ggml_tensor * dev) {
    const ggml_cuda_ec_entry * e = ggml_cuda_ec_find(dev);
    return e ? e->C : 0;
}

static const void * hook_arena(const ggml_tensor * dev) {
    const ggml_cuda_ec_entry * e = ggml_cuda_ec_find(dev);
    return e ? e->arena : nullptr;
}

static const int32_t * hook_ids_slot(const ggml_tensor * dev) {
    const ggml_cuda_ec_entry * e = ggml_cuda_ec_find(dev);
    return e ? e->last_ids : nullptr;
}

int64_t ggml_cuda_ec_ids_n(const ggml_tensor * dev) {
    const ggml_cuda_ec_entry * e = ggml_cuda_ec_find(dev);
    return e ? e->last_ids_n : 0;
}

// [M3-F DUP oracle accessors] input_cpy device base, host slab base, slab stride
void * ggml_cuda_ec_dev_data(const ggml_tensor * dev) {
    const ggml_cuda_ec_entry * e = ggml_cuda_ec_find(dev);
    return e && e->dev_ptr ? e->dev_ptr->data : nullptr;
}
const void * ggml_cuda_ec_host_base(const ggml_tensor * dev) {
    const ggml_cuda_ec_entry * e = ggml_cuda_ec_find(dev);
    return e ? e->host_base : nullptr;
}
int64_t ggml_cuda_ec_slab(const ggml_tensor * dev) {
    const ggml_cuda_ec_entry * e = ggml_cuda_ec_find(dev);
    return e ? e->slab : 0;
}

static void hook_fallback_full(const ggml_tensor * dev, void * stream_v) {
    ggml_cuda_ec_entry * e = ggml_cuda_ec_find(dev);
    GGML_ASSERT(e && e->host_base && e->dev_ptr);
    // rare safety net (should never fire with fresh-ids forcing in place): refill input_cpy in full.
    // Sync the stream FIRST: the M3-A ev_wait skip assumes no legacy write to input_cpy is ever
    // in flight while the served path is engaged; this path is the exception that proves it.
    g_ec_fallbacks++;
    // [M3-D] during CUDA graph capture (EC graphs mode) syncs are illegal — record the refill
    // memcpy only. Nothing can be in flight on the capturing stream, and weights are static,
    // so the recorded node is correct at every replay.
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing((cudaStream_t) stream_v, &cap);
    if (cap != cudaStreamCaptureStatusNone) {
        // capture-safe: no syncs, and the SOURCE must be pinned (pageable sources are illegal
        // under capture). e->pinned is the cudaHostAlloc mirror taken at load; if it's missing,
        // skip the node — weights are static and a later eager call will refill before use.
        const void * src = e->pinned ? e->pinned : e->host_base;
        if (e->pinned) {
            CUDA_CHECK(cudaMemcpyAsync(e->dev_ptr->data, src,
                                       (size_t) e->n_expert * e->slab, cudaMemcpyHostToDevice,
                                       (cudaStream_t) stream_v));
        }
        return;
    }
    cudaStreamSynchronize((cudaStream_t) stream_v);
    CUDA_CHECK(cudaMemcpyAsync(e->dev_ptr->data, e->host_base,
                               (size_t) e->n_expert * e->slab, cudaMemcpyHostToDevice,
                               (cudaStream_t) stream_v));
    CUDA_CHECK(cudaStreamSynchronize((cudaStream_t) stream_v));
}

void ggml_backend_ec_hooks_register(void) {
    // register unconditionally: LLAMA_EXPERT_CACHE is set by common.cpp during model load, which can
    // run AFTER CUDA device enumeration — hook_active() reads the env lazily instead.
    atexit(ec_atexit_dump);

    g_hooks.active            = hook_active;
    g_hooks.is_cuda_buffer    = hook_is_cuda_buffer;
    g_hooks.setup             = hook_setup;
    g_hooks.lookup_or_evict   = hook_lookup_or_evict;
    g_hooks.copy_to_slot      = hook_copy_to_slot;
    g_hooks.fill_ids          = hook_fill_ids;
    g_hooks.served_copy_phase = hook_served;
    g_hooks.engaged_this_call = hook_engaged;
    g_hooks.cache_slots       = hook_slots;
    g_hooks.arena             = hook_arena;
    g_hooks.ids_slot          = hook_ids_slot;
    g_hooks.fallback_full     = hook_fallback_full;
    // [M3-C diag] gpu-bracket hooks left OFF: cudaEventRecord/Query in the split loop turned
    // into sticky CUDA errors surfacing later in quantize_row_q8_1 (unroot-caused; NO_COPY
    // roofline probe answers the same question from the other side — see hook_copy_to_slot).
    g_hooks.gpu_begin         = nullptr;
    g_hooks.gpu_end           = nullptr;
    ggml_backend_ec_hooks_ptr = &g_hooks;
}
