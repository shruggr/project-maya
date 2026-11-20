// src/inkling2/inkling2_lane.hpp - maya-inkling2's CPU lane: the routed experts that are not in VRAM, computed on the
// CPU from their copies in RAM (never the model's file: docs/INKLING2_PORT.md) while the GPU computes the others.  The
// arithmetic is maya-ds4's CPU rows for a GGUF's experts (src/ds4/ds4_iq_cpu.cpp: ggml-cpu's dots, an activation as
// Q8_K) with the SITU activation between the fused gate | up rows and down.  The thread pool is maya-hy4's (hy4::Lane).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "hy4_lane.hpp"

namespace inkling2 {

/// A layer's experts in RAM: expert of RAM slot r at base + r blob, its fused gate | up matrix (2 n_ff rows of n_embd)
/// at 0, its down matrix (n_embd rows of n_ff) at tr_down.
struct CpuExperts {
    strata::ds4::gguf::IqCpu f;
    const uint8_t* base = nullptr;
    size_t blob = 0, tr_down = 0;
    float beta = 4.0f, lb = 25.0f;
    bool init(int tg, int td, int n_embd, int n_ff, std::string& err);
};

struct CpuEntry {
    int slot = 0;   // the expert's RAM slot
    float w = 0.0f;
};

/// out [n_embd] = sum over the entries of w down(situ(gate x, up x)) (no entries: zeros); x [n_embd]
void cpu_experts(hy4::Lane& lane, const CpuExperts& ex, const std::vector<CpuEntry>& entries, const float* x, float* out);

}  // namespace inkling2
