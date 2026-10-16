#pragma once
// M2-lite expert slot cache (CUDA). Registry of per-MoE-tensor device arenas holding C resident
// expert slabs; decode runs mmvq against the arena using a SLOT-INDEXED copy of the ids tensor.
//
// Keyed by the HOST expert tensor pointer (stable); a secondary index maps the current
// scheduler input_cpy (device-side copy, rotates through sched copy slots) back to the entry.
//
// Consistency model: the scheduler copy phase and the mul_mat_id kernel launch share the SAME
// backend stream, so every cudaMemcpyAsync issued during the copy phase is stream-ordered ahead of
// the GEMM's reads. Arena slots and ids-ring entries may be reused by later calls without racing
// an earlier GEMM (all reuse is queued after it on the same stream). No host-blocking syncs on the
// hot path. in_flight events are ADVISORY ONLY (better eviction victims).
#include <cstdint>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

struct ggml_cuda_ec_slot {
    int32_t     expert_id = -1;    // expert currently resident (-1 = never used)
    uint64_t    last_use  = 0;     // LRU tick (call generation)
    cudaEvent_t in_flight = nullptr; // advisory: copy recorded for this slot
    uint64_t    stamp_gen = 0;     // generation of the call that last touched this slot
};

#define GGML_CUDA_EC_ID_RING 4

struct ggml_cuda_ec_entry {
    const ggml_tensor * host_ptr = nullptr;  // registry key: host expert tensor (stable)
    const ggml_tensor * dev_ptr  = nullptr;  // this call's device-side copy (input_cpy), may rotate
    const void   * host_base = nullptr;  // GGUF mmap base (file-backed; NOT DMA-able fast)
    void         * pinned    = nullptr;  // cudaHostAlloc mirror used as the copy source (see .cu)
    size_t         slab      = 0;        // nb[2] bytes per expert
    int64_t        n_expert  = 0;
    int64_t        C         = 0;
    uint8_t      * arena     = nullptr;  // device arena: C * slab
    std::vector<ggml_cuda_ec_slot> slots;
    std::vector<int64_t> slot_of_id;     // expert_id -> slot, -1 if not resident
    int        engaged = 0;              // decode-sized call: intercept must serve from arena
    int        partial_fill = 0;         // copy phase skipped slabs -> input_cpy only partially valid
    int64_t    ids_expect = 0;           // ggml_nelements() of the op's live ids this call
    uint64_t   gen     = 0;              // call generation (stamp counter base)

    // slot-indexed ids ring: fill_ids uploads this call's ids with expert ids mapped to SLOT indices,
    // same ne/nb layout as the real ids tensor. Ring reuse is stream-ordered behind any GEMM still
    // reading the older entry.
    int32_t *  ids_ring[GGML_CUDA_EC_ID_RING] = {};
    int64_t    ids_ring_cap_ints = 0;
    int        ids_ring_next = 0;
    const int32_t * last_ids   = nullptr;  // most recent uploaded buffer (intercept hands to kernel)
    int64_t         last_ids_n = 0;
};

int ggml_cuda_ec_active(void);
ggml_cuda_ec_entry * ggml_cuda_ec_find(const ggml_tensor * dev);
int64_t ggml_cuda_ec_ids_n(const ggml_tensor * dev);
