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
} ggml_backend_ec_hooks;

// defined in ggml-backend.cpp (NULL default => legacy path); set by CUDA backend init when the
// registry is enabled. Both TUs live inside libggml, so no export attribute is needed.
extern ggml_backend_ec_hooks * ggml_backend_ec_hooks_ptr;

// called by CUDA backend init when the registry is enabled
void ggml_backend_ec_hooks_register(void);
GGML_BACKEND_API int ggml_cuda_ec_active(void);
// element-count of the slot ids uploaded for this call's device tensor (0 if none) — used by the
// mul_mat_id intercept to assert the upload matches the live ids tensor
GGML_BACKEND_API int64_t ggml_cuda_ec_ids_n(const ggml_tensor * dev);
