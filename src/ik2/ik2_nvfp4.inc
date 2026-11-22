// src/ik2/ik2_nvfp4.inc - the NVFP4 block decodes (ggml's NVFP4: 64 values a block, a UE4M3 scale a 16, E2M1 codes),
// device code both builds of maya-ik2 compile: the NVIDIA one (src/ik2/ik2_kernels.cu) and the AMD one, whose
// experts run in maya-ds4's kernels (src/ds4/ds4_iq.cu).  Included inside the includer's namespace, after its helpers
// bits_to_float and u32_to_f2 (two F16 in a word as float2).  Moved here unchanged from ik2_kernels.cu.

__host__ __device__ __forceinline__ float ue4m3(uint8_t x) {   // ggml_ue4m3_to_fp32 (branchless bit-ops)
    const uint32_t e = (x >> 3) & 0xFu;
    const uint32_t m = x & 7u;
    const float norm_f = bits_to_float(((e + 119u) << 23) | (m << 20));
    const float sub_f = (float) m * 0x1.0p-10f;
    const float v = (e == 0u) ? sub_f : norm_f;
    return (x == 0x7Fu) ? 0.0f : v;
}

// Fast register E2M1 pair decode via FP16 high-byte LUT in __byte_perm -> __half22float2:
// FP16 representations of {0, 1, 2, 3, 4, 6, 8, 12} all have low byte 0x00 and high bytes {0x00, 0x3c, 0x40, 0x42, 0x44, 0x46, 0x48, 0x4a}.
__device__ __forceinline__ float2 e2m1_pair_lo(uint32_t s) {
    const uint32_t sel = (s << 4) & 0x7070u;
    const uint32_t sgn = ((s & 8u) << 12) | ((s & 0x800u) << 20);
    return u32_to_f2(__byte_perm(0x42403c00u, 0x4a484644u, sel) | sgn);
}
__device__ __forceinline__ float2 e2m1_pair_hi(uint32_t s) {
    const uint32_t sel = s & 0x7070u;
    const uint32_t sgn = ((s & 0x80u) << 8) | ((s & 0x8000u) << 16);
    return u32_to_f2(__byte_perm(0x42403c00u, 0x4a484644u, sel) | sgn);
}

__device__ __forceinline__ void decode_nvfp4_wv(uint32_t q0, uint32_t q1, float* wv) {
    const float2 w01 = e2m1_pair_lo(q0), w23 = e2m1_pair_lo(q0 >> 16);
    const float2 w45 = e2m1_pair_lo(q1), w67 = e2m1_pair_lo(q1 >> 16);
    const float2 w89 = e2m1_pair_hi(q0), wAB = e2m1_pair_hi(q0 >> 16);
    const float2 wCD = e2m1_pair_hi(q1), wEF = e2m1_pair_hi(q1 >> 16);
    wv[0] = w01.x, wv[1] = w01.y, wv[2] = w23.x, wv[3] = w23.y;
    wv[4] = w45.x, wv[5] = w45.y, wv[6] = w67.x, wv[7] = w67.y;
    wv[8] = w89.x, wv[9] = w89.y, wv[10] = wAB.x, wv[11] = wAB.y;
    wv[12] = wCD.x, wv[13] = wCD.y, wv[14] = wEF.x, wv[15] = wEF.y;
}
