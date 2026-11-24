// tests/ik2_lane_l3_bench.cpp - maya-ik2's CPU lane (cpu_experts_rows) on synthetic experts of Inkling 2 Small's shape
// (gate | up IQ2_XS 2 x 2304 x 4608, down IQ3_XXS 4608 x 2304; no GPU, no model): a call's time with its experts cold
// (a pool larger than the caches, drawn at random) against the same call after cpu_experts_prefetch of its experts, and
// a hash of the outputs (the prefetch and the pieces' threads must not change them).  Run with HY4_LANE_L3=1 to group
// the lane's threads by L3.
//
//   ik2_lane_l3_bench [--calls 200] [--pool 256] [--experts 4] [--rows 3] [--picks 5]
#include "../src/ik2/hip/ik2_hip_lane.hpp"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}
void fill(uint8_t* w, size_t bytes, int type, std::mt19937_64& rng) {
    for (size_t o = 0; o + 8 <= bytes; o += 8) {
        const uint64_t v = rng();
        std::memcpy(w + o, &v, 8);
    }
    const size_t bb = ggml_type_size((ggml_type) type);
    for (size_t o = 0; o + bb <= bytes; o += bb) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.0005f + 0.0001f * (float) (rng() % 64));
        std::memcpy(w + o, &d, 2);
    }
}
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}  // namespace

int main(int argc, char** argv) {
    int calls = 200, pool = 256, nexp = 4, rows = 3, picks = 5;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        const int v = std::atoi(argv[i + 1]);
        if (a == "--calls") calls = v;
        else if (a == "--pool") pool = v;
        else if (a == "--experts") nexp = v;
        else if (a == "--rows") rows = v;
        else if (a == "--picks") picks = v;
    }
    const int E = 4608, F = 2304;
    ik2hip::CpuExperts ex;
    std::string err;
    if (!ex.init(GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS, E, F, err)) {
        std::fprintf(stderr, "init: %s\n", err.c_str());
        return 1;
    }
    const size_t gu_bytes = (size_t) 2 * F * ex.f.gu_row, dn_bytes = (size_t) E * ex.f.d_row;
    ex.tr_down = (gu_bytes + 4095) / 4096 * 4096;
    ex.blob = (ex.tr_down + dn_bytes + 4095) / 4096 * 4096;
    std::vector<uint8_t> mem((size_t) pool * ex.blob);
    std::mt19937_64 rng(7);
    for (int e = 0; e < pool; ++e) {
        fill(mem.data() + (size_t) e * ex.blob, gu_bytes, GGML_TYPE_IQ2_XS, rng);
        fill(mem.data() + (size_t) e * ex.blob + ex.tr_down, dn_bytes, GGML_TYPE_IQ3_XXS, rng);
    }
    ex.base = mem.data();
    hy4::Lane lane(-1);
    std::printf("lane %d threads + caller, %d groups; expert %.2f MB; pool %d (%.1f GB); %d experts, %d rows, %d picks a call\n",
                lane.threads(), lane.nodes(), ex.blob / 1e6, pool, (double) pool * ex.blob / 1e9, nexp, rows, picks);
    std::vector<float> x((size_t) rows * E), out((size_t) rows * E);
    for (float& v : x) v = (float) ((int) (rng() % 2001) - 1000) / 1000.0f;
    // the calls: each its experts (distinct, at random) and its entries (row by row, each a pick of the call's experts)
    struct Call {
        std::vector<int> slots;
        std::vector<ik2hip::CpuRowEntry> ent;
    };
    std::vector<Call> cs((size_t) calls);
    for (Call& c : cs) {
        while ((int) c.slots.size() < nexp) {
            const int e = (int) (rng() % (uint64_t) pool);
            if (std::find(c.slots.begin(), c.slots.end(), e) == c.slots.end()) c.slots.push_back(e);
        }
        for (int k = 0; k < picks; ++k) {
            const int row = k * rows / picks;
            const int slot = c.slots[(size_t) (k < nexp ? k : (int) (rng() % (uint64_t) nexp))];
            c.ent.push_back({row, slot, 0.1f + 0.01f * (float) k});
        }
        std::stable_sort(c.ent.begin(), c.ent.end(), [](auto& a, auto& b) { return a.row < b.row; });
    }
    for (int w = 0; w < 5; ++w) ik2hip::cpu_experts_rows(lane, ex, cs[(size_t) w].ent, x.data(), rows, out.data());
    // cold: each call straight away; warm: its prefetch, then the call (timed alone); hot: the call again
    double t_cold = 0, t_pf = 0, t_warm = 0, t_hot = 0;
    uint64_t h_cold = 0, h_warm = 0;
    for (const Call& c : cs) {
        double t0 = now();
        ik2hip::cpu_experts_rows(lane, ex, c.ent, x.data(), rows, out.data());
        t_cold += now() - t0;
        h_cold = fnv(out.data(), out.size() * 4, h_cold);
    }
    for (const Call& c : cs) {
        double t0 = now();
        ik2hip::cpu_experts_prefetch(lane, ex, c.slots.data(), (int) c.slots.size(), nullptr, 0);
        double t1 = now();
        ik2hip::cpu_experts_rows(lane, ex, c.ent, x.data(), rows, out.data());
        double t2 = now();
        h_warm = fnv(out.data(), out.size() * 4, h_warm);
        ik2hip::cpu_experts_rows(lane, ex, c.ent, x.data(), rows, out.data());
        t_hot += now() - t2;
        t_pf += t1 - t0;
        t_warm += t2 - t1;
    }
    const double mb = (double) nexp * ex.blob / 1e6;
    std::printf("us a call: cold %.1f (%.0f GB/s) | prefetch %.1f (%.0f GB/s), then the call %.1f | again %.1f\n",
                1e6 * t_cold / calls, mb * calls / t_cold / 1e3, 1e6 * t_pf / calls, mb * calls / t_pf / 1e3,
                1e6 * t_warm / calls, 1e6 * t_hot / calls);
    std::printf("hash cold %016llx warm %016llx %s\n", (unsigned long long) h_cold, (unsigned long long) h_warm,
                h_cold == h_warm ? "SAME" : "DIFFERENT");
    return 0;
}
