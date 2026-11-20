// tests/inkling2_dspark_test.cpp - maya-inkling2's drafter alone (src/inkling2/inkling2_dspark) against a reference
// run of llama.cpp PR #32218 (tools/inkling2/ref/inkling-dspark-eval.cpp; docs/INKLING2_DSPARK.md, "Measured"):
//   <ref>/tokens.txt             the prompt's token count, then the prompt's and the target's greedy tokens
//   <ref>/tap-slot<id>.bin       F32 [positions][n_embd]: the target's taps (ids 5, 11, ..., 41, 48)
//   <ref>/drafts-<variant>.txt   a line a generated position n: `n kept`, then 7 x `token:probability`
// The prompt's taps are fed, then for every reference line the taps of position n - 1 and a block drafted after the
// token at n: the 7 drafts and their top probabilities against the reference's.
//
//   inkling2-dspark-test --index <maya-inkling2-dspark-q8_0> --target-pack <maya-inkling2> --ref <dir>
//                        [--variant q8_0] [--gpu 3] [--joint-index <maya-inkling2-dspark-bf16>] [--max-ctx 16384]
#include "inkling2_dspark.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    inkling2::DsparkOptions o;
    std::string ref, variant = "q8_0";
    o.gpu = 0;
    o.profile = true;
    o.verbose = true;
    o.cum_min = 0.0f;
    bool timing_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--index") o.index = val();
        else if (a == "--target-pack") o.target_pack = val();
        else if (a == "--joint-index") o.joint_index = val();
        else if (a == "--ref") ref = val();
        else if (a == "--variant") variant = val();
        else if (a == "--gpu") o.gpu = std::atoi(val().c_str());
        else if (a == "--max-ctx") o.max_ctx = std::atoll(val().c_str());
        else if (a == "--no-profile") o.profile = false;
        else if (a == "--timing-only") timing_only = true;
        else if (a == "--cum") o.cum_min = (float) std::atof(val().c_str());
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (o.index.empty() || o.target_pack.empty() || ref.empty()) {
        std::fprintf(stderr, "usage: inkling2-dspark-test --index DIR --target-pack DIR --ref DIR [--variant q8_0] [--gpu N]\n");
        return 2;
    }
    std::string err;
    inkling2::Dspark d;
    if (!d.load(o, err)) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    const int E = d.n_embd(), NT = inkling2::Dspark::kTaps;
    // the reference
    std::vector<int> tok;
    int n_prompt = 0;
    {
        std::ifstream f(ref + "/tokens.txt");
        if (!(f >> n_prompt)) {
            std::fprintf(stderr, "%s/tokens.txt: cannot read\n", ref.c_str());
            return 1;
        }
        for (int t; f >> t;) tok.push_back(t);
    }
    const int n_all = (int) tok.size();
    std::vector<float> taps((size_t) n_all * NT * E);
    for (int k = 0; k < NT; ++k) {
        const std::string p = ref + "/tap-slot" + std::to_string(d.target_layers()[(size_t) k]) + ".bin";
        std::ifstream f(p, std::ios::binary);
        std::vector<float> one((size_t) n_all * E);
        f.read((char*) one.data(), (std::streamsize) (one.size() * 4));
        if (f.gcount() != (std::streamsize) (one.size() * 4)) {
            std::fprintf(stderr, "%s: not %d rows of %d\n", p.c_str(), n_all, E);
            return 1;
        }
        for (int r = 0; r < n_all; ++r)
            std::copy(one.begin() + (size_t) r * E, one.begin() + (size_t) (r + 1) * E, taps.begin() + ((size_t) r * NT + k) * E);
    }
    struct Line {
        int n, kept, d[7];
        float p[7];
    };
    std::vector<Line> lines;
    {
        std::ifstream f(ref + "/drafts-" + variant + ".txt");
        for (std::string line; std::getline(f, line);) {
            std::istringstream is(line);
            Line l{};
            is >> l.n >> l.kept;
            bool ok = true;
            for (int i = 0; i < 7; ++i) {
                std::string x;
                ok = ok && (bool) (is >> x);
                const size_t c = x.find(':');
                if (!ok || c == std::string::npos) {
                    ok = false;
                    break;
                }
                l.d[i] = std::atoi(x.substr(0, c).c_str());
                l.p[i] = (float) std::atof(x.substr(c + 1).c_str());
            }
            if (ok) lines.push_back(l);
        }
        if (lines.empty()) {
            std::fprintf(stderr, "%s/drafts-%s.txt: no lines\n", ref.c_str(), variant.c_str());
            return 1;
        }
    }
    double t0 = now_ms();
    if (!d.feed(taps.data(), n_prompt, 0, err)) {
        std::fprintf(stderr, "feed: %s\n", err.c_str());
        return 1;
    }
    std::printf("fed %d prompt positions in %.1f ms (%.3f ms a position)\n", n_prompt, now_ms() - t0, (now_ms() - t0) / n_prompt);
    long long same_tok = 0, n_tok = 0, same_block = 0, same_kept = 0, ours_kept = 0, ref_kept = 0, flips_close = 0;
    long long reach_o[7] = {0}, kept_o[7] = {0}, first_diff[7] = {0};
    double perr_sum = 0, perr_max = 0;
    long long perr_n = 0;
    std::vector<int> dt;
    std::vector<float> dp;
    const inkling2::DsparkStats& st = d.stats();
    const double feed0 = st.feed_ms;
    for (size_t li = 0; li < lines.size(); ++li) {
        const Line& l = lines[li];
        if (l.n < n_prompt || l.n < (int) d.end() || l.n + 7 > n_all) continue;
        if (d.end() < l.n && !d.feed(taps.data() + (size_t) d.end() * NT * E, l.n - (int) d.end(), d.end(), err)) {
            std::fprintf(stderr, "feed: %s\n", err.c_str());
            return 1;
        }
        if (!d.draft(tok[(size_t) l.n], 7, dt, dp, err)) {
            std::fprintf(stderr, "draft: %s\n", err.c_str());
            return 1;
        }
        if (timing_only) continue;
        if (dt.size() != 7) {
            std::fprintf(stderr, "draft at %d: %zu tokens\n", l.n, dt.size());
            return 1;
        }
        if (timing_only) continue;
        // the drafts after the first difference follow another chain: only the common prefix and the first differing
        // draft compare like with like
        int common = 0;
        while (common < 7 && dt[(size_t) common] == l.d[common]) ++common;
        for (int i = 0; i < 7; ++i) {
            ++n_tok;
            if (dt[(size_t) i] == l.d[i]) ++same_tok;
        }
        for (int i = 0; i < std::min(7, common + 1); ++i) {
            if (i < common) {
                const double e = std::fabs((double) dp[(size_t) i] - l.p[i]);
                perr_sum += e;
                perr_max = std::max(perr_max, e);
                ++perr_n;
            }
        }
        if (common == 7) ++same_block;
        else {
            ++first_diff[common];
            if (l.p[common] < 0.6f) ++flips_close;   // (the reference's own top probability there: a near tie)
            if (first_diff[0] + first_diff[1] + first_diff[2] + first_diff[3] + first_diff[4] + first_diff[5] + first_diff[6] <= 12)
                std::printf("  n %d draft %d: ours %d (p %.3f), the reference %d (p %.3f)\n", l.n, common + 1, dt[(size_t) common],
                            dp[(size_t) common], l.d[common], l.p[common]);
        }
        int mk = 0;
        while (mk < 7 && l.n + 1 + mk < n_all && dt[(size_t) mk] == tok[(size_t) (l.n + 1 + mk)]) ++mk;
        ours_kept += mk;
        ref_kept += l.kept;
        if (mk == l.kept) ++same_kept;
        for (int i = 0; i < 7 && i <= mk; ++i) {
            ++reach_o[i];
            if (i < mk) ++kept_o[i];
        }
    }
    const long long nb = st.drafts;
    std::printf("%lld blocks; a block %.2f ms (embed %.2f, layers %.2f, base logits %.2f, head %.2f); feeding a position %.2f ms\n",
                nb, st.draft_ms / nb, st.embed_ms / nb, st.layers_ms / nb, st.base_ms / nb, st.head_ms / nb,
                (st.feed_ms - feed0) / std::max<long long>(1, st.fed - n_prompt));
    std::printf("layers: Q/K/V %.2f, attention %.2f, its output %.2f, FFN %.2f ms a block\n", st.qkv_ms / nb, st.attn_ms / nb,
                st.o_ms / nb, st.ffn_ms / nb);
    std::printf("VRAM %.2f GB, load %.1f s\n", st.vram_gb, st.load_s);
    if (!timing_only) {
        long long nd = 0;
        for (int i = 0; i < 7; ++i) nd += first_diff[i];
        std::printf("blocks with the reference's 7 drafts: %lld of %lld (%.1f%%); drafts equal position by position: %lld of %lld\n",
                    same_block, nb, 100.0 * same_block / nb, same_tok, n_tok);
        std::printf("first differing draft by position:");
        for (int i = 0; i < 7; ++i) std::printf(" %lld", first_diff[i]);
        std::printf("; %lld of the %lld where the reference's own top probability was under 0.6\n", flips_close, nd);
        std::printf("top probability against the reference (drafts on the common chain): mean |diff| %.4f, max %.4f (%lld)\n",
                    perr_n ? perr_sum / perr_n : 0.0, perr_max, perr_n);
        std::printf("kept by the target's tokens: ours %.3f a block, the reference %.3f; the same count in %lld of %lld blocks\n",
                    (double) ours_kept / nb, (double) ref_kept / nb, same_kept, nb);
        std::printf("ours, kept / reached by position:");
        for (int i = 0; i < 7; ++i) std::printf(" %lld/%lld", kept_o[i], reach_o[i]);
        std::printf("\n");
    }
    return 0;
}
