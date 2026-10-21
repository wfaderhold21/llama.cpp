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
#include <fcntl.h>    // [disk source] open
#include <unistd.h>   // [disk source] pread

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
// [M4-PF] opportunistic prefetch diagnostics: copies issued / bytes / windows skipped (inflight gate).
static std::atomic<uint64_t> g_pf_issued{0}, g_pf_bytes{0}, g_pf_skip{0};
// [M5-BC] batched-copy diagnostics.
static std::atomic<uint64_t> g_bc_batches{0}, g_bc_copies{0};
    // [M6-RR] cache-conditioned routing (Strata STRATA_ROUTE_RESIDENT steal, MIT). Margin in
    // SELECTION-PROB space (Nemotron gates with sigmoid): swap only when the router's own prob
    // for the resident stand-in is within margin of the missed expert's prob.
    static const float g_rr_margin = [] {
        const char * v = getenv("LLAMA_EC_ROUTE_RESIDENT");
        return v ? strtof(v, nullptr) : 0.0f;                    // 0 = off (ship default)
    }();
    static std::atomic<uint64_t> g_rr_offered{0}, g_rr_taken{0}, g_rr_applied{0};
    static const bool g_rr_dbg = getenv("LLAMA_EC_RR_DEBUG") != nullptr;
// [M5-BC] batched-copy A/B knob: default off (per-copy fork/join) until it wins an A/B.
static const bool g_batchcopy_on = getenv("LLAMA_EC_BATCHCOPY") != nullptr;

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

// [steal P1, PolyStrata] usage-profile seed/save: an earlier run's routing counts, kept in a
// file, seed this run's per-expert frequency counters — so hot-set preheat ranks experts from
// real traffic BEFORE seeing any, exactly like PolyStrata's expert-profile.txt ("its profile is
// a file of this install's own: it starts as the counts the repository ships" and the engine
// "saves the experts' use after its first answer"). Counts keep accumulating during the run and
// the file is rewritten at exit. Env: LLAMA_EC_PROFILE=<path>. atexit save only: kill -9 loses
// the run's counts but never corrupts the file (temp+rename).
static void ec_profile_seed(ggml_cuda_ec_entry & e, const char * path);
static void ec_profile_save(const char * path);

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
                printf("[expert-cache] route_resident margin=%.3f offered=%llu swapped=%llu applied=%llu\n",
            g_rr_margin, (unsigned long long) g_rr_offered.load(),
            (unsigned long long) g_rr_taken.load(), (unsigned long long) g_rr_applied.load());
    printf("[expert-cache] batchcopy batches=%llu copies=%llu\n",
            (unsigned long long) g_bc_batches.load(),
            (unsigned long long) g_bc_copies.load());
    printf("[expert-cache] prefetch issued=%llu bytes=%lluMB skipped_busy=%llu\n",
            (unsigned long long) g_pf_issued.load(),
            (unsigned long long) (g_pf_bytes.load() >> 20),
            (unsigned long long) g_pf_skip.load());
    printf("[expert-cache] intercepts=%llu declines=%llu setups=%llu fillids=%llu gate_noserve=%llu graph_launches=%llu\n",
            (unsigned long long) g_ec_intercepts.load(), (unsigned long long) g_ec_declines.load(),
            (unsigned long long) g_ec_setups.load(), (unsigned long long) g_ec_fillids.load(),
            (unsigned long long) g_ec_gate_noserve.load(), (unsigned long long) g_ec_graph_launches.load());
    // [steal P1] persist routing counts for the next run's warm-start (forward-declared above)
    if (ec_n_cached() > 0) {
        const char * pp = getenv("LLAMA_EC_PROFILE");
        if (pp && *pp) ec_profile_save(pp);
    }
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

// [steal P1, PolyStrata] profile file: one line per expert tensor —
//   <tensor name>\t<n_expert>\t<count_0> <count_1> ...
// Written temp+rename at exit; seed is best-effort (a missing/short/garbage line is skipped,
// the run just learns its own hot set like before).
static void ec_profile_seed(ggml_cuda_ec_entry & e, const char * path) {
    if (!e.host_ptr || !e.host_ptr->name) return;
    FILE * f = fopen(path, "r");
    if (!f) return;
    char * line = nullptr; size_t cap = 0;
    while (getline(&line, &cap, f) != -1) {
        char * tab1 = strchr(line, '\t');
        if (!tab1) continue;
        *tab1 = '\0';
        if (strcmp(line, e.host_ptr->name) != 0) continue;
        char * p = tab1 + 1;
        long long n = strtoll(p, &p, 10);
        if (n != (long long) e.n_expert || *p != '\t') continue;
        p++;
        int idx = 0;
        while (idx < e.n_expert) {
            char * endp = nullptr;
            unsigned long long c = strtoull(p, &endp, 10);
            if (endp == p) break;                        // malformed: keep what we seeded
            e.freq[idx++] = c;
            p = endp;
            if (*p == ' ') p++;
            else break;
        }
        if (idx == e.n_expert) {
            fprintf(stderr, "[expert-cache] profile: seeded %lld freq counters from %s\n",
                    (long long) e.n_expert, path);
        }
        break;
    }
    free(line); fclose(f);
}

static void ec_profile_save(const char * path) {
    char tmp[4096];
    if (snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, (int) getpid()) >= (int) sizeof(tmp)) return;
    FILE * f = fopen(tmp, "w");
    if (!f) return;
    fprintf(f, "EC-PROFILE v1\n");
    for (auto & kv : g_entries) {
        const ggml_cuda_ec_entry & e = kv.second;
        if (!e.host_ptr || !e.host_ptr->name || e.freq.empty()) continue;
        fprintf(f, "%s\t%lld\t", e.host_ptr->name, (long long) e.n_expert);
        for (size_t i = 0; i < e.freq.size(); i++) fprintf(f, "%llu%c",
                (unsigned long long) e.freq[i], i + 1 == e.freq.size() ? '\n' : ' ');
    }
    fclose(f);
    if (rename(tmp, path) != 0) remove(tmp);
}
static std::unordered_map<const ggml_tensor *, ggml_cuda_ec_entry *> g_dev_idx;

// [disk source] true when the HOST expert tensor has a loader-registered GGUF range AND the
// entry for it opened its fd + staging ring successfully. Consulted by the scheduler's legacy
// copy branch (input tensor) — if the ring allocation failed we must NOT claim disk, or the
// legacy path would leave stale slabs; the entry falls back to mapping reads at create time.
static int hook_disk(const ggml_tensor * host) {
    auto it = g_entries.find(host);
    return (it != g_entries.end() && it->second.disk && it->second.fd >= 0 &&
            it->second.dstage[0]) ? 1 : 0;
}

static int hook_is_gpu_buffer(const ggml_tensor * t) {
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

static ggml_cuda_ec_entry * lookup_or_create(ggml_backend_t backend, const ggml_tensor * host, const ggml_tensor * dev) {
    // [dual-GPU] sporc is a 4080 Super + 5060 Ti box: stream/event handles are per-device-context,
    // and the sched thread's "current device" is not guaranteed to be the split backend's (crashed:
    // cudaEventRecord invalid resource handle, current device 1 vs backend device 0). Pin the thread
    // to the backend's device for the whole creation, record it, and re-pin in the copy helpers.
    if (backend) {
        ggml_cuda_set_device(((ggml_backend_cuda_context *) backend->context)->device);
    }
    auto it = g_entries.find(host);
    if (it != g_entries.end()) {
        ggml_cuda_ec_entry * e = &it->second;
        // this call's device-side copy is (possibly) a different sched slot — re-index it
        e->dev_ptr = dev;
        g_dev_idx[dev] = e;
        return e;
    }
    GGML_ASSERT(host->buffer && !hook_is_gpu_buffer(host)); // entry mirrors HOST weights, not device

    ggml_cuda_ec_entry e;
    e.host_ptr   = host;
    e.dev_ptr    = dev;
    e.host_base  = host->data; // cudaHostAlloc'd pinned base
    e.slab       = host->nb[2];
    e.n_expert   = host->ne[2];
    e.C          = ec_n_cached();
    if (backend) {
        e.device = ((ggml_backend_cuda_context *) backend->context)->device;
    }

    // [disk source] if the loader registered this tensor's GGUF range (and disk source is on),
    // open a private O_RDONLY fd and size the pinned staging ring. The mapping (host_base)
    // remains a valid cold fallback — its pages demand-fault from the same file — but no hot
    // path touches it, which is the bounded-RSS guarantee on unified-memory machines.
    {
        const char * src = getenv("LLAMA_EXPERT_CACHE_SOURCE");
        const bool want_disk = src && strcmp(src, "disk") == 0;
        ggml_ec_file_src fs;
        if (want_disk && host->name && ggml_backend_ec_get_file_source(host->name, &fs)) {
            int fd = ::open(fs.path, O_RDONLY);
            if (fd < 0) {
                fprintf(stderr, "[expert-cache] disk source: open(%s) failed -> mapping fallback for %s\n",
                        fs.path, host->name);
            } else {
                e.disk      = true;
                e.fd        = fd;
                e.file_offs = fs.offs;
                for (int i = 0; i < 4; i++) {
                    if (cudaHostAlloc(&e.dstage[i], e.slab, cudaHostAllocDefault) != cudaSuccess) {
                        e.dstage[i] = nullptr; // ring degrades: hook_copy_to_slot falls back
                    } else {
                        CUDA_CHECK(cudaEventCreateWithFlags(&e.dstage_done[i], cudaEventDisableTiming));
                    }
                }
                static std::atomic<int> n_disk{0};
                if (n_disk.fetch_add(1) == 0) {
                    fprintf(stderr, "[expert-cache] disk source ACTIVE (pread %s @%llu, slab=%.1fMB)\n",
                            fs.path, (unsigned long long) fs.offs, e.slab / 1e6);
                }
            }
        }
    }
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
    {   // [steal P1] seed freq from a previous run's saved profile (PolyStrata-style warm LRU)
        const char * pp = getenv("LLAMA_EC_PROFILE");
        if (pp && *pp) ec_profile_seed(e, pp);
    }
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
    // [disk source] never mirror in disk mode: cudaHostAlloc(region) would materialize ALL experts
    // in RAM (~21 GB) — the exact thing disk mode exists to prevent.
    if (!e.disk && getenv("LLAMA_EC_PINNED_MIRROR")) {
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
    if (getenv("LLAMA_EC_COPY_STREAMS") && strtoll(getenv("LLAMA_EC_COPY_STREAMS"), nullptr, 10) >= 2) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&e.copy_stream2, cudaStreamNonBlocking));  // [M5-CS]
        CUDA_CHECK(cudaEventCreateWithFlags(&e.batch_join2, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&e.fork_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&e.batch_join, cudaEventDisableTiming));  // [M5-BC]

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
    ggml_cuda_ec_entry * e = lookup_or_create(backend, host, dev);
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
    // [M5-BC] open the copy-batch window HERE: the authoritative miss copies are issued by the
    // ggml-backend.cpp prefetch-hint loop between setup() and fill_ids() (copy_to_slot), NOT by
    // fill_ids own miss loop — opening at fill_ids meant the window never covered a real copy
    // (batches=0 on the first A/B). Closed + joined at fill_ids end. Decline paths leave the
    // window open harmlessly: join-waits on an un-recorded batch_join complete immediately and
    // the next setup re-opens; stale loading slots still join via the per-slot fallback loop.
    e->in_batch     = g_batchcopy_on;
    e->batch_slots.clear();
    // [M6-RR] per-call rr bookkeeping + capture probe: never swap while the stream is
    // capturing (a captured copy would BAKE swapped bytes into every replay).
    e->rr_applied   = false;
    e->rr_w_host    = nullptr;
    e->rr_w_cpy     = nullptr;
    e->rr_w_bytes   = 0;
    {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream_for(backend), &cap);
        e->rr_capturing = (cap != cudaStreamCaptureStatusNone);
    }
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

static int ec_copy_to_slot_impl(ggml_cuda_ec_entry * e, int32_t id, int64_t slot, cudaStream_t kstream) {
    GGML_ASSERT(e && e->host_base && slot >= 0 && slot < e->C);
    ggml_cuda_set_device(e->device);  // [dual-GPU] stream/event handles belong to this device
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
    // [disk source] miss fill from the GGUF: pread slab into the next free pinned staging slot
    // (host blocks only on the NVMe read itself, ~2-4 ms at slab=3.7MB), then true async DMA on
    // the copy stream. Staging ring is 4-deep; reuse gated on the recorded DMA-completion event.
    // Mapping pages are never faulted, so expert RSS stays ~0 — the Mac requirement.
    if (e->disk && e->fd >= 0) {
        int s = e->dstage_next;
        while (e->dstage_done[s] && cudaEventQuery(e->dstage_done[s]) == cudaErrorNotReady) {
            s = (s + 1) & 3;
            if (s == e->dstage_next) {
                // ring fully in flight: oldest is guaranteed to finish; bounded wait, no busy-spin
                if (e->dstage_done[s]) CUDA_CHECK(cudaEventSynchronize(e->dstage_done[s]));
                break;
            }
        }
        e->dstage_next = (s + 1) & 3;
        void * stage = e->dstage[s];
        if (!stage) goto mapping_fallback;  // staging cudaHostAlloc failed at create: slow-but-correct
        {
            const ssize_t got = ::pread(e->fd, stage, e->slab,
                                        (off_t) (e->file_offs + (uint64_t) id * e->slab));
            if (got != (ssize_t) e->slab) {
                fprintf(stderr, "[expert-cache] disk pread short (%zd/%zu) id=%d -> mapping fallback\n",
                        got, e->slab, id);
                goto mapping_fallback;
            }
        }
        CUDA_CHECK(cudaEventRecord(e->fork_event, kstream));
        CUDA_CHECK(cudaStreamWaitEvent(e->copy_stream, e->fork_event, 0));
        CUDA_CHECK(cudaMemcpyAsync(dst, stage, e->slab, cudaMemcpyHostToDevice, e->copy_stream));
        if (!sl.in_flight) {
            CUDA_CHECK(cudaEventCreateWithFlags(&sl.in_flight, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(sl.in_flight, e->copy_stream));
        if (e->dstage_done[s]) CUDA_CHECK(cudaEventRecord(e->dstage_done[s], e->copy_stream));
        sl.loading = true;
        e->side_copies++;
        e->side_bytes += e->slab;
        g_ec_copy_bytes += e->slab;
        g_ec_copy_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count();
        return 1;
    }
mapping_fallback:;
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
        // [M5-BC] LLAMA_EC_BATCHCOPY=1: inside a fill_ids batch window the fork is taken ONCE
        // (first copy of the window) and the join is a single shared event recorded at window
        // end - host API cost amortized across the batch instead of 4+ calls per copy.
        if (g_batchcopy_on && e->in_batch) {
            if (e->batch_slots.empty()) {
                CUDA_CHECK(cudaEventRecord(e->fork_event, kstream));
                CUDA_CHECK(cudaStreamWaitEvent(e->copy_stream, e->fork_event, 0));
                if (e->copy_stream2) CUDA_CHECK(cudaStreamWaitEvent(e->copy_stream2, e->fork_event, 0));
                g_bc_batches++;
            }
            // [M5-CS] stream affinity: slots that already copied keep their stream (same-stream
            // ordering makes overwriting an in-flight copy safe); new slots split round-robin.
            if (!sl.cs && e->copy_stream2 && (e->cs_rr++ & 1)) sl.cs = 1;
            cudaStream_t cs_stream = (sl.cs && e->copy_stream2) ? e->copy_stream2 : e->copy_stream;
            CUDA_CHECK(cudaMemcpyAsync(dst, src, e->slab, cudaMemcpyHostToDevice, cs_stream));
            sl.in_flight = sl.cs ? e->batch_join2 : e->batch_join;  // shared join per stream
            sl.loading   = true;
            e->batch_slots.push_back(slot);
            g_bc_copies++;
        } else {
            CUDA_CHECK(cudaEventRecord(e->fork_event, kstream));
            CUDA_CHECK(cudaStreamWaitEvent(e->copy_stream, e->fork_event, 0));
            if (!sl.cs && e->copy_stream2 && (e->cs_rr++ & 1)) sl.cs = 1;  // [M5-CS] affinity
            cudaStream_t cs_stream = (sl.cs && e->copy_stream2) ? e->copy_stream2 : e->copy_stream;
            CUDA_CHECK(cudaStreamWaitEvent(cs_stream, e->fork_event, 0));
            CUDA_CHECK(cudaMemcpyAsync(dst, src, e->slab, cudaMemcpyHostToDevice, cs_stream));
            if (!sl.in_flight) {
                CUDA_CHECK(cudaEventCreateWithFlags(&sl.in_flight, cudaEventDisableTiming));
            }
            CUDA_CHECK(cudaEventRecord(sl.in_flight, cs_stream));
        }
        sl.loading = true; // cleared when fill_ids joins the kernel stream onto this slot
        e->side_copies++;
        e->side_bytes += e->slab;
    }
    g_ec_copy_bytes += e->slab;
    g_ec_copy_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();
    return 1;
}

static int hook_copy_to_slot(ggml_backend_t backend, const ggml_tensor * dev, int32_t id, int64_t slot) {
    ggml_cuda_ec_entry * e = ggml_cuda_ec_find(dev);
    if (!e) return 0;
    return ec_copy_to_slot_impl(e, id, slot, stream_for(backend));
}

// [disk source] shared body for scheduler-legacy fills: stream input_cpy slabs [first_id,
// last_id] from the GGUF through the pinned staging ring onto kstream. Same ordering contract
// as the stock set_async-from-host it replaces (see hook_copy_slabs comment).
static void ec_fill_dev_slabs(ggml_cuda_ec_entry * e, const ggml_tensor * dev,
                              int32_t first_id, int32_t last_id, cudaStream_t kstream) {
    ggml_cuda_set_device(e->device);  // [dual-GPU] staging_done[] events belong to this device
    uint8_t * dst = (uint8_t *) dev->data;
    for (int32_t id = first_id; id <= last_id; id++) {
        int s = e->dstage_next;
        while (e->dstage_done[s] && cudaEventQuery(e->dstage_done[s]) == cudaErrorNotReady) {
            s = (s + 1) & 3;
            if (s == e->dstage_next) {
                if (e->dstage_done[s]) CUDA_CHECK(cudaEventSynchronize(e->dstage_done[s]));
                break;
            }
        }
        e->dstage_next = (s + 1) & 3;
        void * stage = e->dstage[s];
        if (!stage) continue;  // staging cudaHostAlloc failed at create: log-once happened there;
                               // stale slab here would be data corruption -> caller declines instead
                               // (hook_disk() returns false when staging ring is empty)
        const ssize_t got = ::pread(e->fd, stage, e->slab,
                                    (off_t) (e->file_offs + (uint64_t) id * e->slab));
        if (got != (ssize_t) e->slab) { continue; }  // short read: same policy (unreachable in practice)
        CUDA_CHECK(cudaMemcpyAsync(dst + (size_t) id * e->slab, stage, e->slab,
                                   cudaMemcpyHostToDevice, kstream));
        if (e->dstage_done[s]) CUDA_CHECK(cudaEventRecord(e->dstage_done[s], kstream));
    }
}

// [disk source] scheduler legacy-copy replacement (ec_on==0 MoE branch, ggml-backend.cpp):
// fill input_cpy slab [id] from the GGUF. Must obey the SAME ordering contract the stock
// set_async-from-host had: enqueued on the split backend's kernel stream so later GEMM reads
// are stream-ordered behind it — do NOT route through the arena side-stream (nothing would
// join it for this tensor; the arena path's join lives in fill_ids, which the legacy branch
// never calls). Pinned staging keeps the DMA truly async; ring slot reuse is gated on the
// event recorded after its enqueue on this same stream.
static void hook_copy_slabs(ggml_backend_t backend, ggml_tensor * dev, int32_t first_id, int32_t last_id) {
    ggml_cuda_ec_entry * e = ggml_cuda_ec_find(dev);
    if (!e || !e->disk || e->fd < 0) {
        return;  // defensive: caller only routes here when disk(host) said yes
    }
    ec_fill_dev_slabs(e, dev, first_id, last_id, stream_for(backend));
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

static void ec_prefetch_run(struct ggml_cuda_ec_entry * e, cudaStream_t kstream);

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
    const bool bc_used = e->in_batch && !e->batch_slots.empty();  // [M5-BC]
    if (bc_used) {
        bool has1 = false;                                        // [M5-CS] any copy on stream 2?
        for (int64_t bs : e->batch_slots) if (e->slots[bs].cs) { has1 = true; break; }
        CUDA_CHECK(cudaEventRecord(e->batch_join, e->copy_stream));
        CUDA_CHECK(cudaStreamWaitEvent(stream_for(backend), e->batch_join, 0));
        if (has1) {
            CUDA_CHECK(cudaEventRecord(e->batch_join2, e->copy_stream2));
            CUDA_CHECK(cudaStreamWaitEvent(stream_for(backend), e->batch_join2, 0));
        }
        for (int64_t bs : e->batch_slots) e->slots[bs].loading = false;
    }
    e->in_batch = false;
    for (int i = 0; i < n; i++) {
        const int32_t s = stage[i];
        ggml_cuda_ec_slot & sl = e->slots[s];
        if (!sl.loading || !sl.in_flight) continue;
        if (bc_used && (sl.in_flight == e->batch_join || sl.in_flight == e->batch_join2)) continue;
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
    // [M6-RR] per-token served proof: fill_ids accepting exactly ids_expect slot ids means the
    // ring the (captured or live) mmvq reads is valid THIS token. hook_engaged only runs at graph
    // CAPTURE time, so under replay it can never be the gate.
    if (e->ids_expect == 0 || (int64_t) n == e->ids_expect) e->n_served++;
    g_ec_fillids++;
    // [M4-PF] opportunistic prefetch after the join (host-side only; capture legality handled
    // inside via event-query pressure gate — cudaEventQuery IS legal mid-capture? NO: guard).
    if (cap_stage == cudaStreamCaptureStatusNone) {
        ec_prefetch_run(e, stream_for(backend));
    }
    return 1;
}

// [M4-PF] opportunistic prefetch (end of fill_ids, windowed). Roofline: LLAMA_EC_NO_COPY probe =
// 119.7 t/s vs 71.5 real (Oct 19) -> decode is copy-bound and the DMA copy engines sit below
// line rate. Rank this entry's NON-resident experts by routing frequency (freq window closes
// early = steady routing distribution; preheat already pinned the hottest ~K — prefetch targets
// the tail) and copy them into arena slots on copy_stream under the SAME reservation rules a
// real miss uses (empty > settled-LRU > hot/inflight fallback), with two deltas:
//   - never steal a slot stamped THIS call (its GEMM launch reads it);
//   - never stamp our own copy afterwards: a same-token reader is legal via fork/join ordering,
//     and leaving it unstamped keeps the slot evictable if the copy is slow to land.
// Claim (slot_of_id + loading) is set AT ISSUE so a real miss on that id resolves as a hit and
// fill_ids' join waits on the slot's in_flight event. Capture-unsafe APIs avoided; window rate
// + inflight-pressure gate bound host issue cost. Env LLAMA_EC_PREFETCH=<gens-per-window>, 0=off.
#define EC_PF_COPIES 8
static void ec_prefetch_run(ggml_cuda_ec_entry * e, cudaStream_t kstream) {
    static const long long pf_rate = [] {
        const char * v = getenv("LLAMA_EC_PREFETCH");
        return v ? strtoll(v, nullptr, 10) : 0LL;
    }();
    if (pf_rate <= 0 || e->C <= 0 || e->n_expert <= 0) return;
    if (e->gen < 64 || (long long) (e->gen % (uint64_t) pf_rate) != 0) return;
    ggml_cuda_set_device(e->device);
    // Skip the window if EC_PF_COPIES copies are already loading (a cudaStreamQuery gate
    // starved v1: the copy stream is never idle at decode speed).
    {
        int inflight = 0;
        for (int64_t s0 = 0; s0 < e->C; s0++) {
            const ggml_cuda_ec_slot & cand = e->slots[s0];
            if (cand.loading && cand.in_flight &&
                cudaEventQuery(cand.in_flight) == cudaErrorNotReady && ++inflight >= EC_PF_COPIES) {
                g_pf_skip++;
                return;
            }
        }
    }
    std::vector<int32_t> order;
    order.reserve((size_t) e->n_expert);
    for (int32_t i = 0; i < (int32_t) e->n_expert; i++) {
        if (e->slot_of_id[i] < 0) order.push_back(i);
    }
    const std::vector<uint64_t> & f = e->freq;
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return f[a] > f[b]; });
    int64_t budget = EC_PF_COPIES;
    for (size_t oi = 0; oi < order.size() && budget > 0; oi++) {
        const int32_t id = order[oi];
        if (f[id] == 0) break;                                   // no routing evidence: stop
        // victim: empty > LRU-oldest unstamped > LRU-oldest (hot/inflight last) — mirrors
        // lookup_or_evict, never touching slots stamped this call.
        int64_t slot = -1;
        for (int pass = 0; pass < 3 && slot < 0; pass++) {
            uint64_t oldest = UINT64_MAX;
            for (int64_t s0 = 0; s0 < e->C; s0++) {
                const ggml_cuda_ec_slot & cand = e->slots[s0];
                if (cand.stamp_gen == e->gen) continue;           // read by this call — untouchable
                if (cand.hot && pass == 0) continue;              // keep hot while cold victims exist
                const bool inflight = cand.in_flight && cudaEventQuery(cand.in_flight) == cudaErrorNotReady;
                if (inflight && pass < 2 && oldest != UINT64_MAX) continue;
                if (cand.expert_id < 0) { slot = s0; break; }
                if (cand.last_use < oldest) { oldest = cand.last_use; slot = s0; }
            }
        }
        if (slot < 0) break;                                     // nothing stealable: stop window
        ggml_cuda_ec_slot & sl = e->slots[slot];
        if (sl.expert_id >= 0) e->slot_of_id[sl.expert_id] = -1; // unbind the victim's occupant
        sl.expert_id = id;                                       // claim BEFORE the copy issues
        sl.last_use  = e->gen;
        sl.hot       = false;
        e->slot_of_id[id] = slot;                                // misses on this id now HIT + join
        ec_copy_to_slot_impl(e, id, slot, kstream);              // fork/join-ordered side-stream DMA
        sl.loading   = true;                                     // cleared by the join in fill_ids
        g_pf_issued++; g_pf_bytes += e->slab;
        budget--;
    }
}


    // --- [M6-RR] cache-conditioned routing hooks (scheduler call sites in ggml-backend.cpp) ---
    static int hook_swap_candidate(const ggml_tensor * dev, int32_t id, const float * probs,
                                   const int32_t * tok_row, int n_used, int32_t * out_new) {
        *out_new = -1;
        if (g_rr_margin <= 0.0f || !probs || !tok_row || n_used <= 0) return 0;
        const ggml_cuda_ec_entry * e = ggml_cuda_ec_find(dev);
        if (!e || !e->engaged || e->C <= 0 || e->n_expert <= 0 || e->n_served == 0 || e->rr_capturing) return 0;
        if (id < 0 || id >= (int32_t) e->n_expert) return 0;
        if (e->slot_of_id[id] >= 0) return 0;                         // resident: nothing to win
        g_rr_offered++;
        const float p_id = probs[id];
        int32_t best = -1; float best_p = -1.0f;
        for (int32_t f = 0; f < (int32_t) e->n_expert; f++) {
            if (e->slot_of_id[f] < 0) continue;                       // must be resident
            bool used = false;                                        // must be un-picked this token
            for (int q = 0; q < n_used; q++) used |= (tok_row[q] == f);
            if (used) continue;
            const float pf = probs[f];
            if (pf > best_p) { best_p = pf; best = f; }
        }
        if (best >= 0 && (double) p_id - (double) best_p <= (double) g_rr_margin) {
            *out_new = best; g_rr_taken++;
            if (g_rr_dbg && g_rr_taken.load() <= 20) {
                fprintf(stderr, "[ec-rr] %s id=%d(p=%.4f) -> %d(p=%.4f) gap=%.4f\n",
                        ggml_get_name(e->host_ptr), id, p_id, best, best_p, (double) p_id - best_p);
            }
            return 1;
        }
        return 0;
    }

    // Snapshot the operand, edit swapped token rows in place from the selection probs, and
    // RE-UPLOAD to the split copy with a fresh async copy enqueued here: stream-ordered after
    // every input copy enqueued so far and before compute, so the device operand ends up
    // carrying the edited bytes whether the scheduler's own copy already fired, fires later
    // from the edited buffer, or was baked at capture (graphs replay). Snapshot powers the
    // fallback_full restore.
    static int hook_rr_apply(ggml_backend_t backend, const ggml_tensor * dev, const float * probs,
                             int64_t probs_stride, const int32_t * ids_orig, const int32_t * ids_new,
                             int n_used, int nt, void * w_host, ggml_tensor * w_cpy, int mode, float scale) {
        if (!probs || !ids_orig || !ids_new || !w_host || n_used <= 0 || n_used > 64 ||
            nt <= 0 || nt > 8 || probs_stride <= 0) return 0;
        ggml_cuda_ec_entry * e = ggml_cuda_ec_find(dev);
        if (!e || e->rr_capturing) return 0;
        const int nu = n_used;
        const size_t bytes = (size_t) n_used * (size_t) nt * sizeof(float);   // w [1,ne,nt] contig
        // validate ALL new ids BEFORE any edit lands (fail closed, buffer untouched)
        int ndiff = 0;
        for (int t = 0; t < nt; t++)
            for (int r = 0; r < nu; r++) {
                const int32_t id = ids_new[(size_t) t * n_used + r];
                if (id < 0 || id >= (int32_t) e->n_expert) return 0;
                ndiff += (ids_orig[(size_t) t * n_used + r] != id);
            }
        if (!ndiff) return 0;
        e->rr_w_snap.resize(bytes / sizeof(float));
        memcpy(e->rr_w_snap.data(), w_host, bytes);
        e->rr_w_host  = w_host;
        e->rr_w_cpy   = w_cpy;
        e->rr_w_bytes = bytes;
        for (int t = 0; t < nt; t++) {
            bool dif = false;
            for (int r = 0; r < nu; r++) dif |= (ids_orig[(size_t) t * n_used + r] != ids_new[(size_t) t * n_used + r]);
            if (!dif) continue;                                     // untouched rows stay byte-original
            float * row = (float *) w_host + (size_t) t * (size_t) n_used;
            double raw[64];
            for (int r = 0; r < nu; r++) {
                const int32_t id = ids_new[(size_t) t * n_used + r];
                raw[r] = (double) probs[(size_t) t * (size_t) probs_stride + (size_t) id];
            }
            if (mode == 1) {
                float den = 0.0f;
                for (int r = 0; r < nu; r++) den += (float) raw[r];
                if (den < 6.103515625e-5f) den = 6.103515625e-5f;   // ffn_moe_weights_sum_clamp
                for (int r = 0; r < nu; r++) row[r] = (float) ((double) (float) raw[r] / (double) den) * scale;
            } else if (mode == 2) {
                double mx = 0.0, ex[64], den = 0.0;
                for (int r = 0; r < nu; r++) if (raw[r] > mx) mx = raw[r];
                for (int r = 0; r < nu; r++) { ex[r] = exp(raw[r] - mx); den += ex[r]; }
                if (!(den > 0.0)) den = 1.0;
                for (int r = 0; r < nu; r++) row[r] = (float) (ex[r] / den) * scale;
            } else {
                for (int r = 0; r < nu; r++) row[r] = (float) raw[r] * scale;
            }
        }
        // belt: fresh copy of the edited operand to the split copy (NULL operand copy = MUL
        // reads the host buffer directly — the scheduler's own copy reads edited bytes anyway).
        if (w_cpy) {
            CUDA_CHECK(cudaMemcpyAsync(w_cpy->data, w_host, bytes, cudaMemcpyHostToDevice, stream_for(backend)));
        }
        g_rr_applied++;
        e->rr_applied = true;
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
    ggml_cuda_ec_entry * e = ggml_cuda_ec_find(dev);  // [M6-RR] non-const: n_served++
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
    const bool rr_served = (e->ids_expect == 0 || e->last_ids_n == e->ids_expect);
    if (rr_served) e->n_served++;   // [M6-RR] prove-the-shape-serves gate for route-resident
    return rr_served ? 1 : 0;
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
    // [M6-RR] the intercept declined a call we swapped weights for: legacy reads the ORIGINAL
    // device ids, so the operand must go byte-original too — restore host buffer from snapshot
    // and re-upload ahead of the refill below (the syncs/streams below still order it pre-GEMM).
    if (e->rr_applied && e->rr_w_host && !e->rr_w_snap.empty()) {
        memcpy(e->rr_w_host, e->rr_w_snap.data(), e->rr_w_bytes);
        if (e->rr_w_cpy && e->rr_w_cpy->data) {
            cudaMemcpyAsync(e->rr_w_cpy->data, e->rr_w_host, e->rr_w_bytes,
                            cudaMemcpyHostToDevice, (cudaStream_t) stream_v);
        }
        e->rr_applied = false;
    }
    // [disk source] refill input_cpy (dev_ptr) from the GGUF via the staging ring on the
    // caller's stream — same target the stock full memcpy below writes, so the legacy GEMM
    // reads a fully-populated tensor; mapping pages stay unfaulted.
    if (e->disk && e->fd >= 0) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing((cudaStream_t) stream_v, &cap);
        if (cap != cudaStreamCaptureStatusNone) {
            return; // capture-time full refill on disk source: not shippable; callers must decline
        }
        CUDA_CHECK(cudaStreamSynchronize((cudaStream_t) stream_v));
        ec_fill_dev_slabs(e, e->dev_ptr, 0, (int32_t) e->n_expert - 1, (cudaStream_t) stream_v);
        CUDA_CHECK(cudaStreamSynchronize((cudaStream_t) stream_v));
        return;
    }
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
    g_hooks.is_gpu_buffer     = hook_is_gpu_buffer;
    g_hooks.disk              = hook_disk;
    g_hooks.copy_slabs        = hook_copy_slabs;
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
    g_hooks.swap_candidate    = hook_swap_candidate;                       // [M6-RR]
    g_hooks.rr_apply          = hook_rr_apply;                             // [M6-RR]
    // [M3-C diag] gpu-bracket hooks left OFF: cudaEventRecord/Query in the split loop turned
    // into sticky CUDA errors surfacing later in quantize_row_q8_1 (unroot-caused; NO_COPY
    // roofline probe answers the same question from the other side — see hook_copy_to_slot).
    g_hooks.gpu_begin         = nullptr;
    g_hooks.gpu_end           = nullptr;
    ggml_backend_ec_hooks_ptr = &g_hooks;
}
