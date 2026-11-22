// src/ik2/ik2_loop.inc - the loop's message gates (Inkling-2-Mini), device code both builds of maya-ik2 compile: the
// NVIDIA one (src/ik2/ik2_kernels.cu, which launches them) and the AMD one (src/ik2/hip/ik2_hip_kernels.cu).
// Included inside the includer's namespace, after its helpers wsum (a warp's sum), bsum (a block's, sh [blockDim.x])
// and kBlk (256).  Moved here unchanged from ik2_kernels.cu.

// The loop's message gates, up to three (a layer's q, k and v inputs) in two launches that spread over the SMs.
// sc [n][T][2 R + 2]: per gate and token, the two halves of each g before the norms (D's x columns . x, its m columns
// . m), then the two sums of squares.
struct LoopGateArgs {
    const float *x[3], *m[3];
    const void *D[3], *F[3], *G[3];
    float* y[3];
};
constexpr int kLoopRankMax = 128;
template <bool Compact> __device__ __forceinline__ float gate_weight(const void* p, size_t i) {
    if constexpr (Compact)
        return __uint_as_float((uint32_t)((const uint16_t*)p)[i] << 16);
    else
        return ((const float*)p)[i];
}

// (1) a block a rank (128 threads): its two dots; rank 0's block also stores the sums of squares
template <bool Compact> __global__ void loop_gate_dots_kernel(LoopGateArgs a, int T, int C, int R, float* sc) {
    __shared__ float sh[4][4];
    const int r = blockIdx.x, j = blockIdx.y, t = blockIdx.z, tid = threadIdx.x, warp = tid >> 5;
    const float* xr = a.x[j] + (size_t) t * C;
    const float* mr = a.m[j] + (size_t) t * C;
    const size_t row = (size_t)r * 2 * C;
    float ax = 0.0f, am = 0.0f, sx = 0.0f, sm = 0.0f;
    for (int c = tid; c < C; c += 128) {
        const float xv = xr[c], mv = mr[c];
        ax += gate_weight<Compact>(a.D[j], row + c) * xv, am += gate_weight<Compact>(a.D[j], row + C + c) * mv;
        sx += xv * xv, sm += mv * mv;
    }
    ax = wsum(ax), am = wsum(am), sx = wsum(sx), sm = wsum(sm);
    if ((tid & 31) == 0) sh[warp][0] = ax, sh[warp][1] = am, sh[warp][2] = sx, sh[warp][3] = sm;
    __syncthreads();
    if (tid == 0) {
        float* o = sc + ((size_t) j * T + t) * (2 * R + 2);
        o[2 * r] = sh[0][0] + sh[1][0] + sh[2][0] + sh[3][0];
        o[2 * r + 1] = sh[0][1] + sh[1][1] + sh[2][1] + sh[3][1];
        if (r == 0) {
            o[2 * R] = sh[0][2] + sh[1][2] + sh[2][2] + sh[3][2];
            o[2 * R + 1] = sh[0][3] + sh[1][3] + sh[2][3] + sh[3][3];
        }
    }
}
// (2) y = (1 - tanh(F g)) x + tanh(G g) m: a block is kBlk channels of a token, a thread a channel; F and G by rank
// ([R][C]), so that a warp's threads read side by side.  y may be x (x's sums were taken in (1)).
template <int Rank, bool Compact>
__global__ void loop_gate_apply_kernel(LoopGateArgs a, int T, int C, int R, float eps, const float* __restrict__ sc) {
    __shared__ float sh_g[kLoopRankMax];
    const int j = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
    const float* o = sc + ((size_t) j * T + t) * (2 * R + 2);
    for (int i = tid; i < R; i += blockDim.x) {
        const float rx = rsqrtf(o[2 * R] / (float) C + eps), rm = rsqrtf(o[2 * R + 1] / (float) C + eps);
        sh_g[i] = rx * o[2 * i] + rm * o[2 * i + 1];
    }
    __syncthreads();
    const int c = (int)blockIdx.x * blockDim.x + tid;
    if (c >= C) return;

    float f = 0.0f, gn = 0.0f;
    for (int r = 0; r < (Rank ? Rank : R); ++r) {
        const float gv = sh_g[r];
        f += gate_weight<Compact>(a.F[j], (size_t)r * C + c) * gv,
            gn += gate_weight<Compact>(a.G[j], (size_t)r * C + c) * gv;
    }
    const size_t i = (size_t) t * C + c;
    a.y[j][i] = (1.0f - tanhf(f)) * a.x[j][i] + tanhf(gn) * a.m[j][i];
}

// The gates of a chunk of tokens: their two products (g = D [rms(x); rms(m)], then F g and G g) are matrix products
// taken outside, and these are what is left.  (1) g [T][R] = a1 / rms(x) + a2 / rms(m), a1 = x D_x^T and
// a2 = m D_m^T; a block a token.
__global__ void loop_gate_mix_kernel(const float* __restrict__ x, const float* __restrict__ m,
                                     const float* __restrict__ a1, const float* __restrict__ a2, int C, int R, float eps,
                                     float* __restrict__ g) {
    __shared__ float sh[kBlk];
    const int t = blockIdx.x, tid = threadIdx.x;
    const float* xr = x + (size_t) t * C;
    const float* mr = m + (size_t) t * C;
    float sx = 0.0f, sm = 0.0f;
    for (int c = tid; c < C; c += kBlk) {
        const float xv = xr[c], mv = mr[c];
        sx += xv * xv, sm += mv * mv;
    }
    sx = bsum(sx, sh), sm = bsum(sm, sh);
    if (tid < R) {
        const float rx = rsqrtf(sx / (float) C + eps), rm = rsqrtf(sm / (float) C + eps);
        g[(size_t) t * R + tid] = rx * a1[(size_t) t * R + tid] + rm * a2[(size_t) t * R + tid];
    }
}
// (2) y = (1 - tanh f) x + tanh(gn) m, f = g F and gn = g G; y may be x
__global__ void loop_gate_blend_kernel(const float* x, const float* m, const float* __restrict__ f,
                                       const float* __restrict__ gn, size_t n, float* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (1.0f - tanhf(f[i])) * x[i] + tanhf(gn[i]) * m[i];
}
