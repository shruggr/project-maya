// src/inkling2/inkling2_kernels.cu - maya-inkling2's own kernels (inkling2_kernels.cuh).  Written for clarity first
// (the decode path's small vectors: one launch a step of the reference graph); the sums are F32 in a fixed order.
#include "inkling2_kernels.cuh"

#include <cuda_fp16.h>

#include "strata/kernels/fresh_load.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace inkling2::k {

namespace {

std::atomic<int> g_err{0};

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "maya-inkling2 kernel %s: %s\n", what, cudaGetErrorString(e));
        g_err.fetch_add(1, std::memory_order_relaxed);
    }
}

// a block's sum of v over its threads (blockDim.x <= 256, a power of two); every thread gets it
__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + o]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

// block s = a state: stat[4 s] = its 1 / rms, stat[4 s + 1 + q] = (state / rms) . query q
__global__ void mix_stat_kernel(const float* __restrict__ blocks, int nb, const float* __restrict__ cur, int E,
                                const float* __restrict__ query, int nq, float eps, float* __restrict__ stat) {
    __shared__ float sh[256];
    const int st = blockIdx.x;
    const float* x = st < nb ? blocks + (size_t) st * E : cur;
    float ss = 0.0f, d[3] = {0.0f, 0.0f, 0.0f};
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        const float v = x[i];
        ss += v * v;
        for (int q = 0; q < nq; ++q) d[q] += v * query[(size_t) q * E + i];
    }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float) E + eps);
    for (int q = 0; q < nq; ++q) {
        const float dq = block_sum(d[q], sh);
        if (threadIdx.x == 0) stat[4 * st + 1 + q] = dq * inv;
    }
    if (threadIdx.x == 0) stat[4 * st] = inv;
}
__global__ void mix_out_kernel(const float* __restrict__ blocks, int nb, const float* __restrict__ cur, int E, int nq,
                               const float* __restrict__ stat, float* __restrict__ out) {
    const int q = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    const int ns = nb + 1;
    const float sc = rsqrtf((float) E);
    float m = -INFINITY;
    for (int st = 0; st < ns; ++st) m = fmaxf(m, stat[4 * st + 1 + q] * sc);
    float den = 0.0f, v = 0.0f;
    for (int st = 0; st < ns; ++st) {
        const float w = expf(stat[4 * st + 1 + q] * sc - m);
        den += w;
        v += w * stat[4 * st] * (st < nb ? blocks[(size_t) st * E + i] : cur[i]);
    }
    out[(size_t) q * E + i] = v / den;
}

__global__ void rms_kernel(const float* x, size_t ldx, int n, const float* __restrict__ w, float eps,
                           float* y, size_t ldy) {
    __shared__ float sh[256];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float* yr = y + (size_t) blockIdx.x * ldy;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * inv * (w ? w[i] : 1.0f);
}

__global__ void conv_kernel(float* __restrict__ y, float* __restrict__ hist, const float* __restrict__ w, int C,
                            const float* __restrict__ dyn, int heads, float scale) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float h0 = hist[c], h1 = hist[(size_t) C + c], h2 = hist[(size_t) 2 * C + c], v = y[c];
    float o = w[4 * c] * h0 + w[4 * c + 1] * h1 + w[4 * c + 2] * h2 + w[4 * c + 3] * v;
    if (dyn) {
        const float* d = dyn + 4 * (c / (C / heads));
        o += d[0] * v + d[1] * h2 + d[2] * h1 + d[3] * h0;
    }
    y[c] = (v + o) * scale;
    hist[c] = h1;
    hist[(size_t) C + c] = h2;
    hist[(size_t) 2 * C + c] = v;
}

__global__ void head_clip_kernel(float* __restrict__ x, int n, const float* __restrict__ scale, float thr, float mul) {
    __shared__ float sh[256];
    const int i = threadIdx.x;
    float* xr = x + (size_t) blockIdx.x * n;
    float v = 0.0f;
    if (i < n) v = xr[i] * (scale ? scale[(size_t) blockIdx.x * n + i] : 1.0f);
    const float norm = sqrtf(block_sum(v * v, sh)) + 1e-6f;
    const float f = fminf(1.0f, thr / norm);
    if (i < n) xr[i] = v * f * mul;
}

// thread = (head, key): its score
__global__ void attn_score_kernel(const float* __restrict__ q, const float* __restrict__ rel,
                                  const float* __restrict__ sink, int H, int Hkv, int D, int n_rel,
                                  const float* __restrict__ rel_bias, int extent, const __half* __restrict__ K, int ring,
                                  long long p, long long lo, int nk, float scale, float* __restrict__ S) {
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t) H * nk) return;
    const int h = (int) (idx / (size_t) nk), j = (int) (idx % (size_t) nk);
    const long long pos = lo + j;
    const size_t slot = (size_t) (ring > 0 ? pos % ring : pos);
    const __half* kr = K + (slot * Hkv + (size_t) (h / (H / Hkv))) * D;
    const float* qr = q + (size_t) h * D;
    float a = 0.0f;
    for (int i = 0; i < D; ++i) a += qr[i] * __half2float(kr[i]);
    a *= scale;
    const long long d = p - pos;
    if (d < extent) {
        const float* b = rel_bias + (size_t) d * n_rel;
        const float* r = rel + (size_t) h * n_rel;
        float bs = 0.0f;
        for (int i = 0; i < n_rel; ++i) bs += r[i] * b[i];
        a += bs;
    }
    S[idx] = a - sink[h];
}
// block = a head: its scores -> the probabilities with the zero sink logit in the sum
__global__ void attn_softmax_kernel(float* __restrict__ S, int nk) {
    __shared__ float sh[256];
    float* sr = S + (size_t) blockIdx.x * nk;
    float m = 0.0f;   // (the sink's logit)
    for (int j = threadIdx.x; j < nk; j += blockDim.x) m = fmaxf(m, sr[j]);
    m = block_max(m, sh);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) {
        const float e = expf(sr[j] - m);
        sr[j] = e;
        sum += e;
    }
    sum = block_sum(sum, sh) + expf(-m);
    const float inv = 1.0f / sum;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) sr[j] *= inv;
}
// block = (a tile of kTile keys, a kv head): the tile's K rows and the kv head's queries in shared memory, then a
// thread a (query head, key) - its score (a thread reading its own K row from the cache took most of a decode step's
// attention time)
constexpr int kTile = 32, kSeg = 128;
__global__ void attn_score_tile_kernel(const float* __restrict__ q, const float* __restrict__ rel,
                                       const float* __restrict__ sink, int H, int Hkv, int D, int n_rel,
                                       const float* __restrict__ rel_bias, int extent, const __half* __restrict__ K, int ring,
                                       long long p, long long lo, int nk, float scale, float* __restrict__ S) {
    extern __shared__ float sm[];   // [kTile][D] keys, then [per][D] queries
    const int g = blockIdx.y, j0 = blockIdx.x * kTile, per = H / Hkv;
    float* sk = sm;
    float* sq = sm + (size_t) kTile * D;
    for (int i = threadIdx.x; i < kTile * D; i += blockDim.x) {
        const int j = j0 + i / D;
        if (j < nk) {
            const long long pos = lo + j;
            sk[i] = __half2float(K[((size_t) (ring > 0 ? pos % ring : pos) * Hkv + g) * D + i % D]);
        }
    }
    for (int i = threadIdx.x; i < per * D; i += blockDim.x) sq[i] = q[(size_t) g * per * D + i];
    __syncthreads();
    for (int e = threadIdx.x; e < per * kTile; e += blockDim.x) {
        const int r = e / kTile, jj = e % kTile, j = j0 + jj, h = g * per + r;
        if (j >= nk) continue;
        float a = 0.0f;
        for (int i = 0; i < D; ++i) a += sq[r * D + i] * sk[jj * D + i];
        a *= scale;
        const long long d = p - (lo + j);
        if (d < extent) {
            const float* bb = rel_bias + (size_t) d * n_rel;
            const float* rr = rel + (size_t) h * n_rel;
            float bs = 0.0f;
            for (int i = 0; i < n_rel; ++i) bs += rr[i] * bb[i];
            a += bs;
        }
        S[(size_t) h * nk + j] = a - sink[h];
    }
}
// block = (a segment of kSeg keys, a kv head): part = the segment's sum of P_j V_j for each of the kv head's query
// heads, a V row read once for them - then the segments' sum
__global__ void attn_part_kernel(const float* __restrict__ S, int H, int Hkv, int D, const __half* __restrict__ V, int ring,
                                 long long lo, int nk, int nseg, float* __restrict__ part) {
    const int g = blockIdx.y, seg = blockIdx.x, i = threadIdx.x, per = H / Hkv;
    if (i >= D || per > 8) return;
    const size_t col = (size_t) g * D + i, row = (size_t) Hkv * D;
    const int j1 = min(nk, (seg + 1) * kSeg);
    float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int j = seg * kSeg; j < j1; ++j) {
        const long long pos = lo + j;
        const float x = __half2float(V[(size_t) (ring > 0 ? pos % ring : pos) * row + col]);
        for (int r = 0; r < per; ++r) v[r] += S[(size_t) (g * per + r) * nk + j] * x;
    }
    for (int r = 0; r < per; ++r) part[((size_t) (g * per + r) * nseg + seg) * D + i] = v[r];
}
__global__ void attn_sum_kernel(const float* __restrict__ part, int D, int nseg, float* __restrict__ out) {
    const int h = blockIdx.x, i = threadIdx.x;
    if (i >= D) return;
    float v = 0.0f;
    for (int seg = 0; seg < nseg; ++seg) v += part[((size_t) h * nseg + seg) * D + i];
    out[(size_t) h * D + i] = v;
}

__global__ void sigmoid_mul_kernel(float* __restrict__ x, const float* __restrict__ g, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 1.0f / (1.0f + expf(-g[i]));
}
__global__ void add_kernel(float* __restrict__ y, const float* __restrict__ x, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}
__global__ void situ_kernel(const float* __restrict__ gu, int ff, float beta, float lb, float* __restrict__ h) {
    const int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ff) return;
    const float g = gu[(size_t) r * 2 * ff + i];
    float u = gu[(size_t) r * 2 * ff + ff + i];
    if (lb > 0.0f) u = lb * tanhf(u / lb);
    h[(size_t) r * ff + i] = beta * tanhf(g / beta) * (1.0f / (1.0f + expf(-g))) * u;
}
__global__ void wsum_kernel(const float* __restrict__ part, int m, int n, const float* __restrict__ w,
                            float* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = 0.0f;
    for (int j = 0; j < m; ++j)
        if (w[j] != 0.0f) v = fmaf(w[j], part[(size_t) j * n + i], v);   // (0: not this side's entry)
    out[i] = v;
}
__global__ void softcap_kernel(float* __restrict__ x, size_t n, float cap) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}
__global__ void argmax_kernel(const float* __restrict__ x, int n, float* __restrict__ out) {
    __shared__ float bv[256];
    __shared__ int bi[256];
    float v = -INFINITY;
    int at = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (x[i] > v) {
            v = x[i];
            at = i;
        }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = at;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) {
            const float v2 = bv[threadIdx.x + o];
            const int i2 = bi[threadIdx.x + o];
            if (v2 > bv[threadIdx.x] || (v2 == bv[threadIdx.x] && i2 < bi[threadIdx.x])) {
                bv[threadIdx.x] = v2;
                bi[threadIdx.x] = i2;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[0] = (float) bi[0];
        out[1] = bv[0];
    }
}

// block = a kv head, 1024 threads: its query heads' softmax (the zero sink logit in the sum), then the output - a
// thread a (segment of the keys, value dim), the segments' sums added in shared memory.  One launch for what were
// three (the softmax, the segments' parts, their sum); the scores stay the tiles' launch, which needs its many blocks
// (a single block a kv head that also computed the scores measured twice the four launches' time).
__global__ void attn_soft_out_kernel(float* __restrict__ S, int H, int Hkv, int D, const __half* __restrict__ V, int ring,
                                     long long lo, int nk, float* __restrict__ out) {
    __shared__ float vr[6144];   // [per][threads]: the heads' reductions side by side (a tree for all, not one a head)
    __shared__ float ps[6144];   // [per][segments][D]
    const int g = blockIdx.x, per = H / Hkv, tid = threadIdx.x, B = blockDim.x;
    for (int r = 0; r < per; ++r) {
        const float* sr = S + (size_t) (g * per + r) * nk;
        float m = 0.0f;   // (the sink's logit)
        for (int j = tid; j < nk; j += B) m = fmaxf(m, sr[j]);
        vr[r * B + tid] = m;
    }
    __syncthreads();
    for (int o = B / 2; o > 0; o >>= 1) {
        if (tid < o)
            for (int r = 0; r < per; ++r) vr[r * B + tid] = fmaxf(vr[r * B + tid], vr[r * B + tid + o]);
        __syncthreads();
    }
    float mx[8];
    for (int r = 0; r < per; ++r) mx[r] = vr[r * B];
    __syncthreads();
    for (int r = 0; r < per; ++r) {
        float* sr = S + (size_t) (g * per + r) * nk;
        float sum = 0.0f;
        for (int j = tid; j < nk; j += B) {
            const float e = expf(sr[j] - mx[r]);
            sr[j] = e;
            sum += e;
        }
        vr[r * B + tid] = sum;
    }
    __syncthreads();
    for (int o = B / 2; o > 0; o >>= 1) {
        if (tid < o)
            for (int r = 0; r < per; ++r) vr[r * B + tid] += vr[r * B + tid + o];
        __syncthreads();
    }
    float inv[8];
    for (int r = 0; r < per; ++r) inv[r] = 1.0f / (vr[r * B] + expf(-mx[r]));
    __syncthreads();
    // (the probabilities stay unnormalised in S: the output takes the factor)
    const int nseg = blockDim.x / D, seg = tid / D, i = tid % D, len = (nk + nseg - 1) / nseg;
    const size_t col = (size_t) g * D + i, row = (size_t) Hkv * D;
    float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int j1 = min(nk, (seg + 1) * len);
    for (int j = seg * len; j < j1; ++j) {
        const long long pos = lo + j;
        const float x = __half2float(V[(size_t) (ring > 0 ? pos % ring : pos) * row + col]);
        for (int r = 0; r < per; ++r) v[r] += S[(size_t) (g * per + r) * nk + j] * x;
    }
    for (int r = 0; r < per; ++r) ps[(r * nseg + seg) * D + i] = v[r] * inv[r];
    __syncthreads();
    if (seg == 0)
        for (int r = 0; r < per; ++r) {
            float o = 0.0f;
            for (int q = 0; q < nseg; ++q) o += ps[(r * nseg + q) * D + i];
            out[(size_t) (g * per + r) * D + i] = o;
        }
}
// Split-K attention.  block = (a tile of TK keys, a kv head): the tile's K rows and the head's queries in shared
// memory, a thread a (query head, key) for the scores, each query head's softmax within the tile (its own max), the
// tile's part of the output - st [H][tiles][2] = (the tile's max, its sum of exp(s - max)), part [H][tiles][D]
__global__ void attn_tile_kernel(const float* __restrict__ q, const float* __restrict__ rel, const float* __restrict__ sink,
                                 int H, int Hkv, int D, int n_rel, const float* __restrict__ rel_bias, int extent,
                                 const __half* __restrict__ K, const __half* __restrict__ V, int ring, long long base,
                                 int d0, int nk, int TK, float scale, float* __restrict__ st,
                                 float* __restrict__ part) {
    extern __shared__ float sm[];   // [TK][D] keys, [per][D] queries, [per][TK] scores / weights, [per] maxima
    const int g = blockIdx.y, tile = blockIdx.x, ntile = gridDim.x, j0 = tile * TK, per = H / Hkv, tid = threadIdx.x;
    float* sk = sm;
    float* sq = sk + (size_t) TK * D;
    float* sw = sq + (size_t) per * D;
    float* smb = sw + (size_t) per * TK;
    for (int i = tid; i < TK * D; i += blockDim.x) {
        const int j = j0 + i / D;
        if (j < nk) {   // (key j's row: base + j, wrapped once - no 64-bit modulo a value, which cost most of the launch)
            long long sl = base + j;
            if (ring > 0 && sl >= ring) sl -= ring;
            sk[i] = __half2float(K[((size_t) sl * Hkv + g) * D + i % D]);
        }
    }
    for (int i = tid; i < per * D; i += blockDim.x) sq[i] = q[(size_t) g * per * D + i];
    __syncthreads();
    for (int e = tid; e < per * TK; e += blockDim.x) {
        const int r = e / TK, jj = e % TK, j = j0 + jj, h = g * per + r;
        float a = -INFINITY;
        if (j < nk) {
            a = 0.0f;
            for (int i = 0; i < D; ++i) a += sq[r * D + i] * sk[jj * D + i];
            a *= scale;
            const int d = d0 - j;   // (the query's position less the key's)
            if (d < extent) {
                const float* bb = rel_bias + (size_t) d * n_rel;
                const float* rr = rel + (size_t) h * n_rel;
                float bs = 0.0f;
                for (int i = 0; i < n_rel; ++i) bs += rr[i] * bb[i];
                a += bs;
            }
            a -= sink[h];
        }
        sw[e] = a;
    }
    __syncthreads();
    if (tid < per) {
        float m = -INFINITY;
        for (int jj = 0; jj < TK; ++jj) m = fmaxf(m, sw[tid * TK + jj]);
        smb[tid] = m;
    }
    __syncthreads();
    for (int e = tid; e < per * TK; e += blockDim.x) sw[e] = j0 + e % TK < nk ? expf(sw[e] - smb[e / TK]) : 0.0f;
    __syncthreads();
    if (tid < per) {
        float sum = 0.0f;
        for (int jj = 0; jj < TK; ++jj) sum += sw[tid * TK + jj];
        float* o = st + ((size_t) (g * per + tid) * ntile + tile) * 2;
        o[0] = smb[tid];
        o[1] = sum;
    }
    if (tid < D && per <= 8) {
        const size_t col = (size_t) g * D + tid, row = (size_t) Hkv * D;
        float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        const int n = min(TK, nk - j0);
        for (int jj = 0; jj < n; ++jj) {
            long long sl = base + j0 + jj;
            if (ring > 0 && sl >= ring) sl -= ring;
            const float x = __half2float(V[(size_t) sl * row + col]);
            for (int r = 0; r < per; ++r) v[r] += sw[r * TK + jj] * x;
        }
        for (int r = 0; r < per; ++r) part[((size_t) (g * per + r) * ntile + tile) * D + tid] = v[r];
    }
}
// block = a query head: the tiles' parts under one maximum (the sink's zero logit in it and in the sum), the gate
__global__ void attn_combine_kernel(const float* __restrict__ st, const float* __restrict__ part, int ntile, int D,
                                    const float* __restrict__ gate, float* __restrict__ out) {
    __shared__ float sh[256];
    const int h = blockIdx.x, tid = threadIdx.x;
    const float* sth = st + (size_t) h * ntile * 2;
    float m = 0.0f;
    for (int b = tid; b < ntile; b += blockDim.x) m = fmaxf(m, sth[2 * b]);
    m = block_max(m, sh);
    float sum = 0.0f;
    for (int b = tid; b < ntile; b += blockDim.x) sum += sth[2 * b + 1] * expf(sth[2 * b] - m);
    sum = block_sum(sum, sh) + expf(-m);
    if (tid >= D) return;
    float v = 0.0f;
    for (int b = 0; b < ntile; ++b) v += part[((size_t) h * ntile + b) * D + tid] * expf(sth[2 * b] - m);
    v /= sum;
    if (gate) v *= 1.0f / (1.0f + expf(-gate[(size_t) h * D + tid]));
    out[(size_t) h * D + tid] = v;
}
// one block: the states' norms and dots, then each query's mix, its own norm and gain
__global__ void mix_rms_kernel(const float* __restrict__ blocks, int nb, const float* __restrict__ cur, int E,
                               const float* __restrict__ query, int nq, const float* __restrict__ g0,
                               const float* __restrict__ g1, const float* __restrict__ g2, float eps,
                               float* __restrict__ out) {
    __shared__ float sh[1024];
    __shared__ float st[kMaxStates * 4];
    const int ns = nb + 1, tid = threadIdx.x;
    for (int s = 0; s < ns; ++s) {
        const float* x = s < nb ? blocks + (size_t) s * E : cur;
        float ss = 0.0f, d[3] = {0.0f, 0.0f, 0.0f};
        for (int i = tid; i < E; i += blockDim.x) {
            const float v = x[i];
            ss += v * v;
            for (int qi = 0; qi < nq; ++qi) d[qi] += v * query[(size_t) qi * E + i];
        }
        ss = block_sum(ss, sh);
        const float inv = rsqrtf(ss / (float) E + 1e-6f);
        for (int qi = 0; qi < nq; ++qi) {
            const float dq = block_sum(d[qi], sh);
            if (tid == 0) st[4 * s + 1 + qi] = dq * inv;
        }
        if (tid == 0) st[4 * s] = inv;
        __syncthreads();
    }
    const float sc = rsqrtf((float) E);
    for (int qi = 0; qi < nq; ++qi) {
        const float* gain = qi == 0 ? g0 : qi == 1 ? g1 : g2;
        float m = -INFINITY, w[kMaxStates], den = 0.0f;
        for (int s = 0; s < ns; ++s) m = fmaxf(m, st[4 * s + 1 + qi] * sc);
        for (int s = 0; s < ns; ++s) {
            w[s] = expf(st[4 * s + 1 + qi] * sc - m);
            den += w[s];
        }
        float vv[8], ss = 0.0f;
        int n = 0;
        for (int i = tid; i < E && n < 8; i += blockDim.x, ++n) {
            float v = 0.0f;
            for (int s = 0; s < ns; ++s) v += w[s] * st[4 * s] * (s < nb ? blocks[(size_t) s * E + i] : cur[i]);
            vv[n] = v / den;
            ss += vv[n] * vv[n];
        }
        ss = block_sum(ss, sh);
        const float inv = rsqrtf(ss / (float) E + eps);
        n = 0;
        for (int i = tid; i < E && n < 8; i += blockDim.x, ++n) out[(size_t) qi * E + i] = vv[n] * inv * (gain ? gain[i] : 1.0f);
    }
}
__global__ void lane_add_kernel(const unsigned* flag, unsigned v, const int* __restrict__ ids, const int* __restrict__ slot,
                                int nu, const float* src, float* __restrict__ y, int n) {
    __shared__ int need;
    if (threadIdx.x == 0) {
        int nd = 0;
        for (int j = 0; j < nu; ++j) nd |= slot[ids[j]] < 0;
        if (nd) {
            const unsigned long long t0 = strata::kernels::clock_ns();
            while ((int) (*(volatile const unsigned*) flag - v) < 0)
                if (strata::kernels::clock_ns() - t0 > 60ull * 1000000000ull) {   // (a minute: the lane is gone)
                    nd = 0;
                    break;
                }
        }
        need = nd;
    }
    __syncthreads();
    __threadfence_system();
    if (!need) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] += *(const volatile float*) (src + i);
}

// ---- the decode step's fused forms ----------------------------------------------------------------------------------
// block = (a kv head, K or V): its D channels' conv, then the clip / the norm, the F16 row
__global__ void kv_finish_kernel(float* __restrict__ ky, float* __restrict__ vy, float* __restrict__ hist_k,
                                 float* __restrict__ hist_v, const float* __restrict__ k_conv,
                                 const float* __restrict__ v_conv, const float* __restrict__ dk,
                                 const float* __restrict__ dv, int Hkv, int D, float thr, float eps,
                                 __half* __restrict__ K, __half* __restrict__ V, size_t row) {
    __shared__ float sh[256];
    const int h = blockIdx.x, isv = blockIdx.y, i = threadIdx.x, C = Hkv * D, c = h * D + i;
    float* y = isv ? vy : ky;
    float* hist = isv ? hist_v : hist_k;
    const float* w = isv ? v_conv : k_conv;
    const float* dyn = isv ? dv : dk;
    float o = 0.0f;
    if (i < D) {
        const float h0 = hist[c], h1 = hist[(size_t) C + c], h2 = hist[(size_t) 2 * C + c], v = y[c];
        o = w[4 * c] * h0 + w[4 * c + 1] * h1 + w[4 * c + 2] * h2 + w[4 * c + 3] * v;
        if (dyn) {
            const float* d = dyn + 4 * h;
            o += d[0] * v + d[1] * h2 + d[2] * h1 + d[3] * h0;
        }
        o = (v + o) * (dyn ? 0.333333343f : 0.44721359f);   // (1 / sqrt(9), 1 / sqrt(5))
        hist[c] = h1;
        hist[(size_t) C + c] = h2;
        hist[(size_t) 2 * C + c] = v;
    }
    const float ss = block_sum(o * o, sh);
    const float f = isv ? rsqrtf(ss / (float) D + eps) : fminf(1.0f, thr / (sqrtf(ss) + 1e-6f));
    if (i < D) {
        o *= f;
        y[c] = o;
        (isv ? V : K)[row * C + c] = __float2half_rn(o);
    }
}
__global__ void clip2_kernel(float* __restrict__ q, int D, const float* __restrict__ qs, float qthr,
                             float* __restrict__ rel, int n_rel, const float* __restrict__ rs, float rthr, float mul) {
    __shared__ float sh[256];
    const int i = threadIdx.x, h = blockIdx.x;
    float* qr = q + (size_t) h * D;
    float* rr = rel + (size_t) h * n_rel;
    const float v = i < D ? qr[i] * qs[(size_t) h * D + i] : 0.0f;
    const float r = i < n_rel ? rr[i] * rs[(size_t) h * n_rel + i] : 0.0f;
    const float fq = fminf(1.0f, qthr / (sqrtf(block_sum(v * v, sh)) + 1e-6f));
    const float fr = fminf(1.0f, rthr / (sqrtf(block_sum(r * r, sh)) + 1e-6f));
    if (i < D) qr[i] = v * fq * mul;
    if (i < n_rel) rr[i] = r * fr * mul;
}
__global__ void conv_add_kernel(float* __restrict__ y, float* __restrict__ hist, const float* __restrict__ w, int C,
                                float* __restrict__ cur) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const float h0 = hist[c], h1 = hist[(size_t) C + c], h2 = hist[(size_t) 2 * C + c], v = y[c];
    const float o = (v + w[4 * c] * h0 + w[4 * c + 1] * h1 + w[4 * c + 2] * h2 + w[4 * c + 3] * v) * 0.44721359f;
    y[c] = o;
    cur[c] += o;
    hist[c] = h1;
    hist[(size_t) C + c] = h2;
    hist[(size_t) 2 * C + c] = v;
}

// ---- the prompt path ------------------------------------------------------------------------------------------------
__global__ void mix_stat_rows_kernel(const float* __restrict__ blocks, size_t ldb, int nb, const float* __restrict__ cur,
                                     int E, const float* __restrict__ query, int nq, float eps, float* __restrict__ stat) {
    __shared__ float sh[256];
    const int st = blockIdx.x, t = blockIdx.y, ns = nb + 1;
    const float* x = (st < nb ? blocks + (size_t) st * ldb : cur) + (size_t) t * E;
    float ss = 0.0f, d[3] = {0.0f, 0.0f, 0.0f};
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        const float v = x[i];
        ss += v * v;
        for (int q = 0; q < nq; ++q) d[q] += v * query[(size_t) q * E + i];
    }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float) E + eps);
    float* o = stat + ((size_t) t * ns + st) * 4;
    for (int q = 0; q < nq; ++q) {
        const float dq = block_sum(d[q], sh);
        if (threadIdx.x == 0) o[1 + q] = dq * inv;
    }
    if (threadIdx.x == 0) o[0] = inv;
}
__global__ void mix_out_rows_kernel(const float* __restrict__ blocks, size_t ldb, int nb, const float* __restrict__ cur,
                                    int T, int E, int nq, const float* __restrict__ stat, float* __restrict__ out) {
    const int q = blockIdx.z, t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    const int ns = nb + 1;
    const float* stt = stat + (size_t) t * ns * 4;
    const float sc = rsqrtf((float) E);
    float m = -INFINITY;
    for (int st = 0; st < ns; ++st) m = fmaxf(m, stt[4 * st + 1 + q] * sc);
    float den = 0.0f, v = 0.0f;
    for (int st = 0; st < ns; ++st) {
        const float w = expf(stt[4 * st + 1 + q] * sc - m);
        den += w;
        v += w * stt[4 * st] * (st < nb ? blocks[(size_t) st * ldb + (size_t) t * E + i] : cur[(size_t) t * E + i]);
    }
    out[((size_t) q * T + t) * E + i] = v / den;
}
__global__ void conv_rows_kernel(const float* __restrict__ y, const float* __restrict__ hist, const float* __restrict__ w,
                                 int C, const float* __restrict__ dyn, int heads, float scale, float* __restrict__ out) {
    const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float in[4];   // (in[k]: the input at t - 3 + k)
    for (int k = 0; k < 4; ++k) {
        const int tt = t - 3 + k;
        in[k] = tt >= 0 ? y[(size_t) tt * C + c] : hist[(size_t) (3 + tt) * C + c];
    }
    float o = w[4 * c] * in[0] + w[4 * c + 1] * in[1] + w[4 * c + 2] * in[2] + w[4 * c + 3] * in[3];
    if (dyn) {
        const float* d = dyn + ((size_t) t * heads + (size_t) (c / (C / heads))) * 4;
        o += d[0] * in[3] + d[1] * in[2] + d[2] * in[1] + d[3] * in[0];
    }
    out[(size_t) t * C + c] = (in[3] + o) * scale;
}
__global__ void conv_hist_kernel(const float* __restrict__ y, int T, float* hist, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    for (int r = 0; r < 3; ++r) {   // (the row T + r of hist's three, then the chunk's)
        const int i = T + r;
        hist[(size_t) r * C + c] = i < 3 ? hist[(size_t) i * C + c] : y[(size_t) (i - 3) * C + c];
    }
}
__global__ void head_clip_rows_kernel(float* __restrict__ x, int H, int n, const float* __restrict__ scale, float thr,
                                      float alpha, float inv_floor, long long p0) {
    __shared__ float sh[256];
    const int i = threadIdx.x, h = blockIdx.x % H, t = blockIdx.x / H;
    float* xr = x + (size_t) blockIdx.x * n;
    float v = 0.0f;
    if (i < n) v = xr[i] * (scale ? scale[(size_t) h * n + i] : 1.0f);
    const float norm = sqrtf(block_sum(v * v, sh)) + 1e-6f;
    const float f = fminf(1.0f, thr / norm);
    const float mul = alpha != 0.0f ? 1.0f + alpha * logf(fmaxf(1.0f, (float) (p0 + t + 1) * inv_floor)) : 1.0f;
    if (i < n) xr[i] = v * f * mul;
}
__global__ void store_rows_kernel(const float* __restrict__ src, int n, __half* __restrict__ cache, int ring, long long p0) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long p = p0 + t;
    cache[(size_t) (ring > 0 ? p % ring : p) * n + i] = __float2half_rn(src[(size_t) t * n + i]);
}
__global__ void perm_heads_kernel(const float* __restrict__ src, float* __restrict__ dst, int T, int Hkv, int per, int D,
                                  int to_p) {
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t) T * Hkv * per * D;
    if (idx >= total) return;
    // idx walks x [T][Hkv][per][D]
    const int d = (int) (idx % D), r = (int) (idx / D % per), g = (int) (idx / ((size_t) D * per) % Hkv);
    const int t = (int) (idx / ((size_t) D * per * Hkv));
    const size_t pi = (((size_t) g * T + t) * per + r) * D + d;
    if (to_p) dst[pi] = src[idx];
    else dst[idx] = src[pi];
}
// block = a query
__global__ void attn_softmax_rows_kernel(float* __restrict__ S, int T, int Hkv, int per, int nk, long long klo,
                                         long long p0, int window, int extent, const float* __restrict__ tbl,
                                         const float* __restrict__ sink, float scale) {
    __shared__ float sh[256];
    const int qi = blockIdx.x, g = qi / (T * per), t = qi % (T * per) / per, r = qi % per;
    const int H = Hkv * per, h = g * per + r;
    float* sr = S + (size_t) qi * nk;
    const float* tb = tbl + ((size_t) t * H + h) * extent;
    const float snk = sink[(size_t) t * H + h];
    const long long p = p0 + t;
    // the keys the token sees: positions [lo, p]
    long long lo = window > 0 ? p - (window - 1) : 0;
    if (lo < klo) lo = klo;
    const int j0 = (int) (lo - klo), j1 = (int) (p - klo);   // (inclusive)
    float m = 0.0f;
    for (int j = j0 + threadIdx.x; j <= j1; j += blockDim.x) {
        const long long d = p - (klo + j);
        const float v = sr[j] * scale + (d < extent ? tb[d] : 0.0f) - snk;
        sr[j] = v;
        m = fmaxf(m, v);
    }
    m = block_max(m, sh);
    float sum = 0.0f;
    for (int j = j0 + threadIdx.x; j <= j1; j += blockDim.x) {
        const float e = expf(sr[j] - m);
        sr[j] = e;
        sum += e;
    }
    sum = block_sum(sum, sh) + expf(-m);
    const float inv = 1.0f / sum;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) sr[j] = j >= j0 && j <= j1 ? sr[j] * inv : 0.0f;
}

__global__ void scatter_rows_kernel(const float* __restrict__ y, const int* __restrict__ row_of, int E,
                                    float* __restrict__ ent) {
    const int e = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x, r = row_of[e];
    if (i < E && r >= 0) ent[(size_t) e * E + i] = y[(size_t) r * E + i];
}
__global__ void combine_rows_kernel(const float* __restrict__ ent, const float* __restrict__ w, int nu, int E,
                                    float* __restrict__ out) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    float v = 0.0f;
    for (int j = 0; j < nu; ++j) v = fmaf(w[(size_t) t * nu + j], ent[((size_t) t * nu + j) * E + i], v);
    out[(size_t) t * E + i] = v;
}

unsigned blocks_of(size_t n, int per = 256) { return (unsigned) ((n + (size_t) per - 1) / (size_t) per); }

}  // namespace

void mix(const float* blocks, int nb, const float* cur, int E, const float* query, int nq, float eps, float* stat,
         float* out, cudaStream_t s) {
    if (nb + 1 > kMaxStates || nq > 3) {
        std::fprintf(stderr, "maya-inkling2 mix: %d states, %d queries\n", nb + 1, nq);
        g_err.fetch_add(1);
        return;
    }
    mix_stat_kernel<<<(unsigned) (nb + 1), 256, 0, s>>>(blocks, nb, cur, E, query, nq, eps, stat);
    mix_out_kernel<<<dim3(blocks_of((size_t) E), (unsigned) nq), 256, 0, s>>>(blocks, nb, cur, E, nq, stat, out);
    check("mix");
}
void rms(const float* x, size_t ldx, int rows, int n, const float* w, float eps, float* y, size_t ldy, cudaStream_t s) {
    rms_kernel<<<(unsigned) rows, 256, 0, s>>>(x, ldx, n, w, eps, y, ldy);
    check("rms");
}
void conv(float* y, float* hist, const float* w, int C, const float* dyn, int heads, cudaStream_t s) {
    conv_kernel<<<blocks_of((size_t) C), 256, 0, s>>>(y, hist, w, C, dyn, heads, 1.0f / std::sqrt(dyn ? 9.0f : 5.0f));
    check("conv");
}
void head_clip(float* x, int H, int n, const float* scale, float thr, float mul, cudaStream_t s) {
    if (n > 256) {
        std::fprintf(stderr, "maya-inkling2 head_clip: rows of %d\n", n);
        g_err.fetch_add(1);
        return;
    }
    head_clip_kernel<<<(unsigned) H, 256, 0, s>>>(x, n, scale, thr, mul);
    check("head_clip");
}
void attention(const float* q, const float* rel, const float* sink, int H, int Hkv, int D, int n_rel,
               const float* rel_bias, int extent, const uint16_t* K, const uint16_t* V, int ring, int64_t p, int64_t lo,
               float scale, float* S, float* out, cudaStream_t s) {
    const int nk = (int) (p - lo + 1);
    const int per = H / Hkv;
    if (per <= 8 && (size_t) (kTile + per) * D * 4 <= 40000)
        attn_score_tile_kernel<<<dim3((unsigned) ((nk + kTile - 1) / kTile), (unsigned) Hkv), 256,
                                 (size_t) (kTile + per) * D * sizeof(float), s>>>(q, rel, sink, H, Hkv, D, n_rel, rel_bias,
                                                                                  extent, (const __half*) K, ring, (long long) p,
                                                                                  (long long) lo, nk, scale, S);
    else
        attn_score_kernel<<<blocks_of((size_t) H * nk), 256, 0, s>>>(q, rel, sink, H, Hkv, D, n_rel, rel_bias, extent,
                                                                      (const __half*) K, ring, (long long) p, (long long) lo, nk, scale, S);
    // (the softmax and the output in a launch while the keys are few: measured, 512 decoded tokens at a short context
    //  34.8 tok/s against 32.5 - and 27.3 against 28.1 after a 2.7k prompt, where the window is full: the three
    //  launches then, as before)
    if (per * 1024 <= 6144 && 1024 % D == 0 && nk <= 640) {
        attn_soft_out_kernel<<<(unsigned) Hkv, 1024, 0, s>>>(S, H, Hkv, D, (const __half*) V, ring, (long long) lo, nk, out);
        check("attention");
        return;
    }
    attn_softmax_kernel<<<(unsigned) H, 256, 0, s>>>(S, nk);
    const int nseg = (nk + kSeg - 1) / kSeg;
    float* part = S + (size_t) H * nk;
    attn_part_kernel<<<dim3((unsigned) nseg, (unsigned) Hkv), 256, 0, s>>>(S, H, Hkv, D, (const __half*) V, ring, (long long) lo, nk, nseg, part);
    attn_sum_kernel<<<(unsigned) H, 256, 0, s>>>(part, D, nseg, out);
    check("attention");
}
namespace {
// keys a tile.  Measured at a full 1024-key window (ms of attention a token over the 48 layers): 64 keys 11.2, 32
// keys 8.9, 16 keys 6.4 - the smaller the tile the faster, against what its work would say.  INKLING2_ATTN_TK=a,b
// (tests): a for up to 2048 keys, b beyond.
int split_tk(int nk) {
    static int t1 = 16, t2 = 16;
    static const bool init = [] {
        if (const char* e = std::getenv("INKLING2_ATTN_TK")) {
            int a = 0, b2 = 0;
            if (std::sscanf(e, "%d,%d", &a, &b2) == 2 && a >= 8 && a <= 64 && b2 >= 16 && b2 <= 64) t1 = a, t2 = b2;
        }
        return true;
    }();
    (void) init;
    return nk <= 2048 ? t1 : t2;
}
}  // namespace
size_t attention_split_floats(int H, int D, int64_t max_keys) {
    const int64_t nk = std::max<int64_t>(max_keys, 2048);
    const int64_t tiles = std::max<int64_t>((2048 + 7) / 8, (nk + 15) / 16);
    return (size_t) H * (size_t) tiles * (size_t) (D + 2);
}
void attention_split(const float* q, const float* rel, const float* sink, int H, int Hkv, int D, int n_rel,
                     const float* rel_bias, int extent, const uint16_t* K, const uint16_t* V, int ring, int64_t p,
                     int64_t lo, float scale, float* S, const float* gate, float* out, cudaStream_t s) {
    const int nk = (int) (p - lo + 1), per = H / Hkv, TK = split_tk(nk), ntile = (nk + TK - 1) / TK;
    if (per > 8 || D > 256) {
        std::fprintf(stderr, "maya-inkling2 attention_split: %d query heads a kv head, rows of %d\n", per, D);
        g_err.fetch_add(1);
        return;
    }
    float* part = S + (size_t) H * ntile * 2;
    attn_tile_kernel<<<dim3((unsigned) ntile, (unsigned) Hkv), 256,
                       ((size_t) TK * D + (size_t) per * D + (size_t) per * TK + 16) * sizeof(float), s>>>(
        q, rel, sink, H, Hkv, D, n_rel, rel_bias, extent, (const __half*) K, (const __half*) V, ring,
        (long long) (ring > 0 ? lo % ring : lo), (int) (p - lo), nk, TK, scale, S, part);
    attn_combine_kernel<<<(unsigned) H, 256, 0, s>>>(S, part, ntile, D, gate, out);
    check("attention_split");
}
void sigmoid_mul(float* x, const float* g, size_t n, cudaStream_t s) {
    sigmoid_mul_kernel<<<blocks_of(n), 256, 0, s>>>(x, g, n);
    check("sigmoid_mul");
}
void add(float* y, const float* x, size_t n, cudaStream_t s) {
    add_kernel<<<blocks_of(n), 256, 0, s>>>(y, x, n);
    check("add");
}
void situ(const float* gu, int rows, int ff, float beta, float lb, float* h, cudaStream_t s) {
    if (rows <= 0) return;
    situ_kernel<<<dim3(blocks_of((size_t) ff), (unsigned) rows), 256, 0, s>>>(gu, ff, beta, lb, h);
    check("situ");
}
void wsum(const float* part, int m, int n, const float* w, float* out, cudaStream_t s) {
    wsum_kernel<<<blocks_of((size_t) n), 256, 0, s>>>(part, m, n, w, out);
    check("wsum");
}
void softcap(float* x, size_t n, float cap, cudaStream_t s) {
    softcap_kernel<<<blocks_of(n), 256, 0, s>>>(x, n, cap);
    check("softcap");
}
void argmax(const float* x, int n, float* out, cudaStream_t s) {
    argmax_kernel<<<1, 256, 0, s>>>(x, n, out);
    check("argmax");
}
void mix_rows(const float* blocks, size_t ldb, int nb, const float* cur, int T, int E, const float* query, int nq,
              float eps, float* stat, float* out, cudaStream_t s) {
    if (nb + 1 > kMaxStates || nq > 3) {
        std::fprintf(stderr, "maya-inkling2 mix: %d states, %d queries\n", nb + 1, nq);
        g_err.fetch_add(1);
        return;
    }
    mix_stat_rows_kernel<<<dim3((unsigned) (nb + 1), (unsigned) T), 256, 0, s>>>(blocks, ldb, nb, cur, E, query, nq, eps, stat);
    mix_out_rows_kernel<<<dim3(blocks_of((size_t) E), (unsigned) T, (unsigned) nq), 256, 0, s>>>(blocks, ldb, nb, cur, T, E,
                                                                                              nq, stat, out);
    check("mix_rows");
}
void conv_rows(const float* y, int T, float* hist, const float* w, int C, const float* dyn, int heads, float* out,
               cudaStream_t s) {
    conv_rows_kernel<<<dim3(blocks_of((size_t) C), (unsigned) T), 256, 0, s>>>(y, hist, w, C, dyn, heads,
                                                                                1.0f / std::sqrt(dyn ? 9.0f : 5.0f), out);
    conv_hist_kernel<<<blocks_of((size_t) C), 256, 0, s>>>(y, T, hist, C);
    check("conv_rows");
}
void head_clip_rows(float* x, int T, int H, int n, const float* scale, float thr, float alpha, double floor, int64_t p0,
                    cudaStream_t s) {
    if (n > 256) {
        g_err.fetch_add(1);
        return;
    }
    head_clip_rows_kernel<<<(unsigned) (T * H), 256, 0, s>>>(x, H, n, scale, thr, alpha, (float) (1.0 / floor), (long long) p0);
    check("head_clip_rows");
}
void store_rows(const float* src, int T, int n, uint16_t* cache, int ring, int64_t p0, cudaStream_t s) {
    store_rows_kernel<<<dim3(blocks_of((size_t) n), (unsigned) T), 256, 0, s>>>(src, n, (__half*) cache, ring, (long long) p0);
    check("store_rows");
}
void perm_heads(const float* src, float* dst, int T, int Hkv, int per, int D, bool to_p, cudaStream_t s) {
    perm_heads_kernel<<<blocks_of((size_t) T * Hkv * per * D), 256, 0, s>>>(src, dst, T, Hkv, per, D, to_p ? 1 : 0);
    check("perm_heads");
}
void attn_softmax_rows(float* S, int T, int Hkv, int per, int nk, int64_t klo, int64_t p0, int window, int extent,
                       const float* tbl, const float* sink, float scale, cudaStream_t s) {
    attn_softmax_rows_kernel<<<(unsigned) (T * Hkv * per), 256, 0, s>>>(S, T, Hkv, per, nk, (long long) klo, (long long) p0,
                                                                         window, extent, tbl, sink, scale);
    check("attn_softmax_rows");
}
void kv_finish(float* ky, float* vy, float* hist_k, float* hist_v, const float* k_conv, const float* v_conv,
               const float* dk, const float* dv, int Hkv, int D, float thr, float eps, uint16_t* K, uint16_t* V, int ring,
               int64_t p, cudaStream_t s) {
    if (D > 256) {
        g_err.fetch_add(1);
        return;
    }
    kv_finish_kernel<<<dim3((unsigned) Hkv, 2), 256, 0, s>>>(ky, vy, hist_k, hist_v, k_conv, v_conv, dk, dv, Hkv, D, thr, eps,
                                                             (__half*) K, (__half*) V, (size_t) (ring > 0 ? p % ring : p));
    check("kv_finish");
}
void clip2(float* q, int H, int D, const float* qs, float qthr, float* rel, int n_rel, const float* rs, float rthr,
           float mul, cudaStream_t s) {
    if (D > 256 || n_rel > 256) {
        g_err.fetch_add(1);
        return;
    }
    clip2_kernel<<<(unsigned) H, 256, 0, s>>>(q, D, qs, qthr, rel, n_rel, rs, rthr, mul);
    check("clip2");
}
void conv_add(float* y, float* hist, const float* w, int C, float* cur, cudaStream_t s) {
    conv_add_kernel<<<blocks_of((size_t) C), 256, 0, s>>>(y, hist, w, C, cur);
    check("conv_add");
}
void scatter_rows(const float* y, const int* row_of, int n, int E, float* ent, cudaStream_t s) {
    for (int e0 = 0; e0 < n; e0 += 32768) {   // (a grid's rows at most)
        const int ne = std::min(32768, n - e0);
        scatter_rows_kernel<<<dim3(blocks_of((size_t) E), (unsigned) ne), 256, 0, s>>>(y, row_of + e0, E, ent + (size_t) e0 * E);
    }
    check("scatter_rows");
}
void combine_rows(const float* ent, const float* w, int T, int nu, int E, float* out, cudaStream_t s) {
    combine_rows_kernel<<<dim3(blocks_of((size_t) E), (unsigned) T), 256, 0, s>>>(ent, w, nu, E, out);
    check("combine_rows");
}
void mix_rms(const float* blocks, int nb, const float* cur, int E, const float* query, int nq, const float* g0,
             const float* g1, const float* g2, float eps, float* out, cudaStream_t s) {
    if (nb + 1 > kMaxStates || nq > 3 || E > 8 * 1024) {
        std::fprintf(stderr, "maya-inkling2 mix_rms: %d states, %d queries, width %d\n", nb + 1, nq, E);
        g_err.fetch_add(1);
        return;
    }
    mix_rms_kernel<<<1, 1024, 0, s>>>(blocks, nb, cur, E, query, nq, g0, g1, g2, eps, out);
    check("mix_rms");
}
void lane_add(const unsigned* flag, unsigned v, const int* ids, const int* slot, int nu, const float* src, float* y, int n,
              cudaStream_t s) {
    lane_add_kernel<<<blocks_of((size_t) n), 256, 0, s>>>(flag, v, ids, slot, nu, src, y, n);
    check("lane_add");
}
int errors() { return g_err.exchange(0); }

}  // namespace inkling2::k
