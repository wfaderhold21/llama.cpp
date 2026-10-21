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
    cudaEvent_t in_flight = nullptr; // [M3-B] REAL gate: miss-copy completion (side stream)
    bool        loading   = false;   // [M3-B] copy issued on copy_stream, event not yet known-done
    int8_t      cs = 0;        // [M5-CS] copy-stream affinity (set at first issue, never changes)
    bool        hot       = false;   // [M3-E] pre-staged hot expert: evict LAST, never before non-hot
    uint64_t    stamp_gen = 0;     // generation of the call that last touched this slot
};

#define GGML_CUDA_EC_ID_RING 4

struct ggml_cuda_ec_entry {
    const ggml_tensor * host_ptr = nullptr;  // registry key: host expert tensor (stable)
    const ggml_tensor * dev_ptr  = nullptr;  // this call's device-side copy (input_cpy), may rotate
    const void   * host_base = nullptr;  // GGUF mmap base (file-backed; NOT DMA-able fast)
    void         * pinned    = nullptr;  // cudaHostAlloc mirror used as the copy source (see .cu)
    // [disk source] loader-registered GGUF range for this tensor (see ggml-ec-hooks.h /
    // ggml-backend.h): when disk=true, hot-path fills stream slabs via pread(e.fd, ...,
    // file_offs + id*slab) through pinned staging, never touching host_base's mapping.
    // host_base stays valid as the (slow, cold) fallback source — pages fault from file.
    int            device    = 0;        // [dual-GPU] CUDA device whose backend created the
                                         // stream/events — APIs against them need this current
    bool           disk      = false;
    int            fd        = -1;       // private O_RDONLY fd, opened at entry create
    uint64_t       file_offs = 0;        // tensor data offset in the GGUF
    void         * dstage[4] = {};       // [disk] pinned pread->DMA staging ring (slab-sized)
    cudaEvent_t    dstage_done[4]  = {}; // [disk] DMA-completion per staging slot (reuse gate)
    int            dstage_next     = 0;
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

    // [M3-D] pinned host staging for the slot-ids upload: graph capture cannot snapshot a
    // memcpy whose source is pageable, and replay must re-read THIS buffer each token (the
    // captured memcpy node reads current contents at replay time — same contract as upstream's
    // graph-safe input uploads). cudaHostAlloc'd once alongside the ring, never freed.
    int32_t * stage_pinned   = nullptr;
    int64_t   stage_cap_ints = 0;

    // [M3-B] dedicated H2D copy stream for miss fills: keeps the big copies OFF the kernel
    // stream so sched event-waits/syncs never drain them; the kernel stream waits only on the
    // per-slot completion event of slots the current call actually reads.
    cudaStream_t copy_stream = nullptr;
    cudaStream_t copy_stream2 = nullptr;   // [M5-CS] second DMA engine (asyncEngineCount=2)
    cudaEvent_t  batch_join2  = nullptr;   // join event for copy_stream2 batches
    uint64_t     cs_rr = 0;                // round-robin counter for NEW-slot stream choice
    cudaEvent_t  fork_event  = nullptr;  // fork: copy_stream <- backend stream ordering
    // [M5-BC] batched copies: fill_ids brackets its miss loop with a batch window; all
    // side-stream copies in the window share ONE join event (recorded once at window end) and
    // the kernel stream waits on it once. Slot.in_flight points at the shared event while its
    // copy belongs to a batch (join loop dedupes on the pointer).
    cudaEvent_t  batch_join    = nullptr;
    bool         in_batch      = false;
    std::vector<int64_t> batch_slots;
    uint64_t     side_copies = 0;        // count/bytes of side-stream copies (diagnostics)
    uint64_t     side_bytes  = 0;

    // [M3-E] hot-set pre-staging (Ferrol, Oct 16): rank experts by observed routing frequency
    // (counted in fill_ids); at warmup trigger, bulk-copy the hottest K into slots and mark
    // them hot so eviction drains cold slots first. LLAMA_EC_PRELOAD_MB selects K = MB/slab.
    std::vector<uint64_t> freq;      // per expert_id: routing hits (decode calls)
    bool           preheated = false;
    uint64_t       preheat_gen = 0;  // trigger generation (set from env at entry create)
    uint64_t       preheat_copied = 0;
    // [M6-RR] route-resident state (per decode call; reset in hook_setup):
    //  n_served      — calls the mul_mat_id intercept PROVED it can serve (swap gate: never
    //                  touch a layer whose host path has never been proven arena-servable).
    //  rr_capturing  — set in hook_setup when the backend stream is capturing: skip all edits
    //                  (a captured copy would BAKE swapped bytes into every replay).
    //  rr_*          — the weights operand edited this call + pre-edit snapshot for the
    //                  fallback_full restore (legacy must see ORIGINAL weights + ORIGINAL ids).
    uint64_t       n_served = 0;
    bool           rr_capturing = false;
    bool           rr_applied = false;
    void *         rr_w_host = nullptr;      // CPU weights operand base (edited region start)
    ggml_tensor *  rr_w_cpy  = nullptr;      // its split-backend copy (device) for re-upload
    size_t         rr_w_bytes = 0;
    std::vector<float> rr_w_snap;            // pre-edit snapshot (rr_w_bytes worth)

};

int ggml_cuda_ec_active(void);
ggml_cuda_ec_entry * ggml_cuda_ec_find(const ggml_tensor * dev);
int64_t ggml_cuda_ec_ids_n(const ggml_tensor * dev);
