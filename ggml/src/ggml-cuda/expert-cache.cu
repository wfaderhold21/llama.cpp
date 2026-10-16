#include "common.cuh"
#include "expert-cache.cuh"

#include "ggml-backend-impl.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <cstdio>
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
void ggml_cuda_ec_count_intercept(void) { g_ec_intercepts++; }
void ggml_cuda_ec_count_decline(void)  { g_ec_declines++; }
void ggml_cuda_ec_count_gate_noserve(void) { g_ec_gate_noserve++; }
static void ec_atexit_dump(void) {
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
    printf("[expert-cache] intercepts=%llu declines=%llu setups=%llu fillids=%llu gate_noserve=%llu\n",
            (unsigned long long) g_ec_intercepts.load(), (unsigned long long) g_ec_declines.load(),
            (unsigned long long) g_ec_setups.load(), (unsigned long long) g_ec_fillids.load(),
            (unsigned long long) g_ec_gate_noserve.load());
}

static ggml_backend_ec_hooks g_hooks;
// note: entries intentionally live for process lifetime (freed with the process)

int ggml_cuda_ec_active(void) {
    return ec_n_cached() > 0;
}

// entries keyed by the HOST expert tensor pointer (stable across calls; input_cpy pointers rotate
// through sched copy slots). dev_idx maps the current device-side copy back to its entry.
static std::unordered_map<const ggml_tensor *, ggml_cuda_ec_entry> g_entries;
static std::unordered_map<const ggml_tensor *, ggml_cuda_ec_entry *> g_dev_idx;

static int hook_is_cuda_buffer(const ggml_tensor * t) {
    if (!t->buffer) {
        return 0;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer));
    return dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
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

    CUDA_CHECK(cudaMalloc((void **) &e.arena, (size_t) e.C * e.slab));
    // zero once so block-quant padding inside a slab is never NaN (MMQ/MMVQ safety)
    CUDA_CHECK(cudaMemset(e.arena, 0, (size_t) e.C * e.slab));

    e.slots.assign(e.C, ggml_cuda_ec_slot{});
    e.slot_of_id.assign(e.n_expert, -1);

    // One-time pinned mirror of the expert slab array. The loader hands us GGUF mmap pages
    // (file-backed): cudaHostRegister REFUSES them (verified by probe), and pageable cudaMemcpyAsync
    // from file pages runs ~13.7 GB/s AND blocks the host thread — that alone capped decode at
    // 8.8 t/s. cudaHostAlloc + a single bulk memcpy at load moves misses to true async pinned DMA
    // at PCIe link speed (~25 GB/s) with zero per-copy host cost.
    {
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


static int hook_setup(ggml_backend_t backend, const ggml_tensor * host, ggml_tensor * dev, int64_t n_tokens, int64_t n_ids_elems) {
    const auto t0 = std::chrono::steady_clock::now();
    struct T { std::atomic<uint64_t> * to; const std::chrono::steady_clock::time_point t0;
               ~T() { ec_time(*to, t0); } } tr { &g_t_setup, t0 };
    g_ec_setups++;
    ggml_cuda_ec_entry * e = lookup_or_create(host, dev);
    e->gen++;
    e->ids_expect = n_ids_elems;
    e->partial_fill = 0;
    // engage ONLY for decode-sized ubatches (mirrors the intercept's ne2 <= MMVQ_MAX_BATCH gate).
    // Prefill keeps the legacy input_cpy copy path byte-identically.
    e->engaged   = (n_tokens <= 8) ? 1 : 0;
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
    // In-flight copies are safe to overwrite (same-stream ordering), but prefer finished ones.
    int64_t victim = -1;
    for (int64_t s = 0; s < e->C; s++) {
        if (e->slots[s].expert_id < 0) { victim = s; break; }
    }
    if (victim < 0) {
        uint64_t oldest = UINT64_MAX;
        for (int64_t s = 0; s < e->C; s++) {
            const ggml_cuda_ec_slot & sl = e->slots[s];
            if (sl.stamp_gen == e->gen) continue; // in use by this call
            const bool inflight = sl.in_flight && cudaEventQuery(sl.in_flight) == cudaErrorNotReady;
            if (inflight && oldest != UINT64_MAX) continue; // prefer settled slots
            if (sl.last_use < oldest) { oldest = sl.last_use; victim = s; }
        }
    }
    GGML_ASSERT(victim >= 0); // impossible for decode-sized calls (C >> 6); loud if assumptions break

    ggml_cuda_ec_slot & sl = e->slots[victim];
    if (sl.expert_id >= 0) {
        e->slot_of_id[sl.expert_id] = -1;
        g_ec_evicts++;
    }
    sl.expert_id   = id;
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
    cudaStream_t stream = stream_for(backend);
    const uint8_t * src = (const uint8_t *) (e->pinned ? e->pinned : e->host_base) + (size_t) id * e->slab;
    uint8_t * dst = e->arena + (size_t) slot * e->slab;
    const auto t0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(dst, src, e->slab, cudaMemcpyHostToDevice, stream));
    g_ec_copy_bytes += e->slab;
    g_ec_copy_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();
    ggml_cuda_ec_slot & sl = e->slots[slot];
    if (!sl.in_flight) {
        CUDA_CHECK(cudaEventCreateWithFlags(&sl.in_flight, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventRecord(sl.in_flight, stream)); // advisory only (eviction hints)
    return 1;
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

    const int slot_idx = e->ids_ring_next;
    e->ids_ring_next = (e->ids_ring_next + 1) % GGML_CUDA_EC_ID_RING;
    int32_t * dst_dev = e->ids_ring[slot_idx];

    // stage in a host scratch (thread-local: no per-call malloc) then async-upload on the stream.
    // AUTHORITATIVE resolution: every id routed by THIS call is resolved here (the sched copy loop
    // is only a prefetch — its bitset can lag when upstream reuses an ids tensor). Resolution here
    // issues the miss copy on the same stream, so it is ordered ahead of the GEMM that reads it.
    static thread_local std::vector<int32_t> stage;
    stage.resize(n);
    for (int i = 0; i < n; i++) {
        const int32_t id = host_ids[i];
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
    // ring reuse is stream-ordered behind any GEMM still reading the older slot
    CUDA_CHECK(cudaMemcpyAsync(dst_dev, stage.data(), (size_t) n * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream_for(backend)));
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

static void hook_fallback_full(const ggml_tensor * dev, void * stream_v) {
    ggml_cuda_ec_entry * e = ggml_cuda_ec_find(dev);
    GGML_ASSERT(e && e->host_base && e->dev_ptr);
    // rare safety net (should never fire with fresh-ids forcing in place): refill input_cpy in full
    g_ec_fallbacks++;
    CUDA_CHECK(cudaMemcpyAsync(e->dev_ptr->data, e->host_base,
                               (size_t) e->n_expert * e->slab, cudaMemcpyHostToDevice,
                               (cudaStream_t) stream_v));
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
    ggml_backend_ec_hooks_ptr = &g_hooks;
}
