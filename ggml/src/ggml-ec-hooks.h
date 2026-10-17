#pragma once
// Expert-slot-cache hooks (M2-lite). ggml-backend.cpp cannot include CUDA headers, so the CUDA
// expert cache registers a small function-pointer table here. NULL pointer => legacy path
// (byte-identical default behavior when --expert-cache 0).
//
// Phase protocol inside the scheduler MoE partial-copy block, per decode call:
//   setup(host, dev, n_tokens)        begin call; engages cache iff n_tokens <= 8 (decode-sized)
//   lookup_or_evict(dev, id, ...)     resolve routed id -> resident slot (copy only on miss)
//   copy_to_slot(dev, id, slot)       pinned host slab -> arena slot (async, backend stream)
//   fill_ids(dev, host_ids, n)        map this call's ids values to slots, upload to the id ring
// then at ggml_cuda_mul_mat_id (decode tier):
//   engaged_this_call(dev) && ids_slot(dev)  -> run mmvq against arena + slot-indexed ids.
//
// The scheduler block calls setup() unconditionally for host-weight mul_mat_id inputs when
// active() && is_cuda_buffer(input_cpy); owns()/the other hooks are only consulted after setup.
#include <stdint.h>
#include <atomic>
#include "ggml.h"
#include "ggml-backend.h"

typedef struct ggml_backend_ec_hooks {
    int (*active)(void);                    // registry enabled (LLAMA_EXPERT_CACHE > 0)
    int (*is_cuda_buffer)(const ggml_tensor * t);  // tensor lives on a CUDA device buffer (cross-TU)

    int (*setup)(ggml_backend_t backend, const ggml_tensor * host, ggml_tensor * dev, int64_t n_tokens, int64_t n_ids_elems);
    int (*lookup_or_evict)(const ggml_tensor * dev, int32_t id, int64_t * out_slot, int * out_need_copy);
    int (*copy_to_slot)(ggml_backend_t backend, const ggml_tensor * dev, int32_t id, int64_t slot);
    // map the n int32 routed ids of this call to slot indices and async-upload to device (ring)
    int (*fill_ids)(ggml_backend_t backend, const ggml_tensor * dev, const int32_t * host_ids, int n);

    // called from ggml_cuda_mul_mat_id (CUDA module; kept uniform via the table anyway):
    int (*served_copy_phase)(const ggml_tensor * dev);  // decode call whose weights came from arena
    int (*engaged_this_call)(const ggml_tensor * dev);  // served AND slot-ids consistent -> intercept
    int64_t (*cache_slots)(const ggml_tensor * dev);     // C
    const void * (*arena)(const ggml_tensor * dev);      // device arena base (C * slab)
    const int32_t * (*ids_slot)(const ggml_tensor * dev); // slot-indexed ids for this call, or NULL
    // safety net: if the intercept cannot serve a call the copy phase engaged for, input_cpy was
    // only partially filled — restore it with a full host->device copy before the legacy GEMM reads
    void (*fallback_full)(const ggml_tensor * dev, void * stream);

    // [M3-C diag] optional GPU busy-time brackets around split compute (NULL = off)
    void (*gpu_begin)(ggml_backend_t backend);
    void (*gpu_end)(ggml_backend_t backend);
} ggml_backend_ec_hooks;

// defined in ggml-backend.cpp (NULL default => legacy path); set by CUDA backend init when the
// registry is enabled. Both TUs live inside libggml, so no export attribute is needed.
extern ggml_backend_ec_hooks * ggml_backend_ec_hooks_ptr;

// [M3-A] host-side sync-site profiler (definitions in ggml-backend.cpp, dump in expert-cache.cu).
// One counter pair per per-token host stall we care about proving/removing.
enum ggml_ec_sync_site {
    GGML_EC_SITE_SPLIT_SYNC = 0,   // cross-split prev-backend sync/event sync
    GGML_EC_SITE_INPUT_SYNC,       // synchronize(input_backend) before ids read
    GGML_EC_SITE_IDS_SYNC,         // get_async(ids) + synchronize(ids_backend)
    GGML_EC_SITE_EV_WAIT,          // event_wait(split_backend, sched event)
    GGML_EC_SITE_EV_SYNC,          // event_synchronize before input->data copy
    GGML_EC_SITE_GRAPH_COMPUTE,    // graph_compute_async host issue time
    GGML_EC_SITE_GRAPH_DRAIN,      // [diag] synchronize(split_backend) right after compute
    GGML_EC_SITE_LAYER_SYNC,       // [diag] stream sync inside the mul_mat_id intercept
    GGML_EC_SITE_SPLIT_WALL,       // [diag] host wall inside the ENTIRE compute_splits loop
    GGML_EC_SITE_SCHED_SYNC,       // [diag] ggml_backend_sched_synchronize (whole-token drain)
    GGML_EC_SITE_GC_GPU,           // [diag] graph_compute_async host issue, GPU split
    GGML_EC_SITE_GC_CPU,           // [diag] graph_compute_async host issue, CPU split
    GGML_EC_SITE_PRE_BUILD,        // [diag] build_splits + bookkeeping before the split loop
    GGML_EC_SITE_SPLIT_PRE,        // [diag] per-split: pre-loop hooks + ev wait before compute
    GGML_EC_SITE_SPLIT_POST,       // [diag] per-split: t_end, ev record, copy-back for next split
    GGML_EC_SITE_HOOK_SETUP,       // [diag] ec->setup call in the loop
    GGML_EC_SITE_HOOK_FILL,        // [diag] ec->fill_ids call in the loop
    GGML_EC_SITE_HOOK_PRECOPY,     // [diag] legacy prefetch loop + authoritative fill work
    GGML_EC_SITE_COPY_FALLBACK,    // [diag] sync-copy fallback path: drains BOTH backends
    GGML_EC_SITE_LEGACY_SET,       // [diag] expert set_async enqueues (EC-served: should be ~0 calls)
    GGML_EC_SITE_INPUTS_TOTAL,     // [diag] whole per-input loop per split
    GGML_EC_SITE_ALLOC_GRAPH,      // [diag] split BUILD + alloc inside sched_graph_compute_async
    GGML_EC_SITE_SCHED_OUTER,      // [diag] whole sched_graph_compute_async (reset+alloc+splits)
    GGML_EC_SITE_IN_PREP,          // [diag] per-input: tensor_copy + ec skip predicate
    GGML_EC_SITE_MOE_BRANCH,       // [diag] per-input: whole MUL_MAT_ID branch (post ev_wait)
    GGML_EC_SITE_REG_COPY,         // [diag] per-input: non-MoE else branch (cpy_tensor_async)
    GGML_EC_SITE_FLAG_INPUT,       // [diag] per-input: FLAG_INPUT branch incl tensor_copy
    GGML_EC_SITE_TENSOR_COPY,      // [diag] tensor_id_copy hash lookup alone (M3-F bisect)
    GGML_EC_SITE_PREDICATE,        // [diag] ec_input_skip_wait predicate alone (M3-F bisect)
    GGML_EC_SITE_COUNT
};
extern std::atomic<uint64_t> g_ec_prof_ns[GGML_EC_SITE_COUNT];
extern std::atomic<uint64_t> g_ec_prof_cnt[GGML_EC_SITE_COUNT];

// called by CUDA backend init when the registry is enabled
void ggml_backend_ec_hooks_register(void);
GGML_BACKEND_API int ggml_cuda_ec_active(void);
// element-count of the slot ids uploaded for this call's device tensor (0 if none) — used by the
// mul_mat_id intercept to assert the upload matches the live ids tensor
GGML_BACKEND_API int64_t ggml_cuda_ec_ids_n(const ggml_tensor * dev);
