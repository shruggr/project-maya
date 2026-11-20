// src/inkling2/inkling2_main.cpp - maya-inkling2: Thinking Machines' Inkling 2 Small on the GPUs (inkling2_model.hpp).
//
//   maya-inkling2 --pack <dir> --tokens 371,151451,... [--n-predict N]   greedy decode from token ids, with timings
//   maya-inkling2 --pack <dir> --parity <reference dir>                  the reference's tokens one at a time, every
//                                                                        tensor of its dump compared (llama.cpp's
//                                                                        graph names: docs/INKLING2_PORT.md)
// This version has the decode path only (a prompt is fed token by token); the --serve protocol comes with the batched
// prompt path.
#include "inkling2_model.hpp"

#include "mm3_os.hpp"
#include "mm3_sample.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <random>
#include <thread>
#include <sstream>
#include <string>
#include <vector>

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::vector<int> int_list(const std::string& s) {
    std::vector<int> v;
    std::string x = s;
    for (char& c : x)
        if (c == ',') c = ' ';
    std::istringstream is(x);
    for (int i; is >> i;) v.push_back(i);
    return v;
}

// One tensor of a reference dump (index.jsonl: step, name, file, type, ne)
struct RefT {
    std::string file;
    bool i32 = false;
    int64_t ne[4] = {1, 1, 1, 1};
    int64_t total() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
};
struct Ref {
    std::string dir;
    std::map<std::string, RefT> t;   // "<step>/<name>"
    std::vector<int> prompt, greedy;
    std::map<std::string, FILE*> open;

    static std::string field(const std::string& line, const std::string& key) {
        const size_t a = line.find("\"" + key + "\":");
        if (a == std::string::npos) return "";
        size_t b = a + key.size() + 3;
        if (line[b] == '"') return line.substr(b + 1, line.find('"', b + 1) - b - 1);
        if (line[b] == '[') return line.substr(b + 1, line.find(']', b) - b - 1);
        return line.substr(b, line.find_first_of(",}", b) - b);
    }
    bool load(const std::string& d, std::string& err) {
        dir = d;
        std::ifstream f(dir + "/index.jsonl"), tk(dir + "/tokens.txt");
        if (!f || !tk) {
            err = dir + ": no index.jsonl / tokens.txt (a reference dump: tools/inkling2/ref)";
            return false;
        }
        for (std::string line; std::getline(f, line);) {
            RefT r;
            r.file = field(line, "file");
            r.i32 = field(line, "type") == "i32";
            const std::vector<int> ne = int_list(field(line, "ne"));
            for (size_t i = 0; i < ne.size() && i < 4; ++i) r.ne[i] = ne[i];
            t[field(line, "step") + "/" + field(line, "name")] = r;
        }
        std::string l1, l2;
        std::getline(tk, l1);
        std::getline(tk, l2);
        prompt = int_list(l1);
        greedy = int_list(l2);
        return !prompt.empty();
    }
    // row `row` of `rows` of the step's tensor (n values of 4 bytes) -> dst; false: not in the dump, or another shape
    bool row(int step, const std::string& name, int64_t rw, int64_t rows, size_t n, void* dst) {
        auto it = t.find(std::to_string(step) + "/" + name);
        if (it == t.end() || it->second.total() != (int64_t) n * rows) return false;
        FILE*& fp = open[it->second.file];
        if (!fp) fp = std::fopen((dir + "/" + it->second.file).c_str(), "rb");
        if (!fp) return false;
        return fseeko(fp, (off_t) (rw * (int64_t) n * 4), SEEK_SET) == 0 && std::fread(dst, 4, n, fp) == n;
    }
};

struct Acc {
    double cos_min = 2.0, cos_sum = 0, rel_max = 0;
    int64_t n = 0, hit = 0, all = 0, pos_diff = 0;
};

// ---- --serve: the line protocol serve/server.py's engine speaks (maya-hy4's, docs/INKLING2_SERVING.md) -------------
// The requests' lines from stdin on a thread of their own: STOP and WRAP are flags the request in hand sees at once
struct Lines {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool eof = false;
    std::atomic<bool> stop{false}, wrap{false};
    void start() {
        std::thread([this] {
            std::string buf;
            char chunk[4096];
            for (;;) {
                const long n = mm3::os::read_stdin(chunk, sizeof chunk);
                if (n <= 0) break;
                buf.append(chunk, (size_t) n);
                for (size_t nl; (nl = buf.find('\n')) != std::string::npos;) {
                    std::string l = buf.substr(0, nl);
                    buf.erase(0, nl + 1);
                    if (!l.empty() && l.back() == '\r') l.pop_back();
                    if (l == "STOP" || l.rfind("STOP ", 0) == 0) {
                        stop.store(true);
                        continue;
                    }
                    if (l == "WRAP") {
                        wrap.store(true);
                        continue;
                    }
                    std::lock_guard<std::mutex> lk(mu);
                    q.push_back(std::move(l));
                    cv.notify_one();
                }
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        }).detach();
    }
    bool next(std::string& out) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        out = std::move(q.front());
        q.pop_front();
        return true;
    }
};

// GEN <max_new> [key=value ...] <id,id,...>: keys this engine does not use are taken and ignored
bool parse_gen(const std::string& line, mm3::sample::Req& r) {
    std::istringstream is(line.substr(4));
    r.top_k = 0;   // (no top-k unless asked)
    if (!(is >> r.max_new)) return false;
    for (std::string tok; is >> tok;) {
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {
            r.ids = int_list(tok);
            break;
        }
        const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
        if (k == "temperature") r.temperature = std::strtof(v.c_str(), nullptr);
        else if (k == "top_p") r.top_p = std::strtof(v.c_str(), nullptr);
        else if (k == "top_k") r.top_k = std::atoi(v.c_str());
        else if (k == "min_p") r.min_p = std::strtof(v.c_str(), nullptr);
        else if (k == "seed") r.seed = std::strtoull(v.c_str(), nullptr, 10);
        else if (k == "penalty_last_n") r.pen_n = std::atoi(v.c_str());
        else if (k == "penalty_repeat") r.pen_repeat = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_freq") r.pen_freq = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_present") r.pen_present = std::strtof(v.c_str(), nullptr);
    }
    return !r.ids.empty();
}

// The next token from a row of logits (the padded vocabulary rows are not in it): greedy without a temperature; else
// mm3's distribution (temperature, top_k, top_p, min_p, the penalties) over the tokens within 40 temperature-nats of
// the top - the rest hold under 1e-12 of the mass - or over the whole row when a penalty may move any of them
int sample(const float* row, int V, const mm3::sample::Req& r, const std::vector<int>& hist, std::mt19937_64& rng) {
    if (mm3::sample::penalised(r)) return mm3::sample::draw(mm3::sample::dist(row, V, r, hist), rng);
    int b = 0;
    for (int i = 1; i < V; ++i)
        if (row[i] > row[b]) b = i;
    if (!(r.temperature > 0.0f)) return b;
    const float cut = row[b] - 40.0f * r.temperature;
    if (!(r.top_k > 0) && !(r.top_p < 1.0f) && !(r.min_p > 0.0f)) {   // (plain sampling: the softmax itself, no sort)
        static thread_local std::vector<float> w;
        w.resize((size_t) V);
        const float it = 1.0f / r.temperature, mx = row[b];
        double z = 0;
        for (int i = 0; i < V; ++i) z += (w[(size_t) i] = row[i] >= cut ? std::exp((row[i] - mx) * it) : 0.0f);
        double u = std::uniform_real_distribution<double>(0.0, z)(rng);
        for (int i = 0; i < V; ++i) {
            u -= w[(size_t) i];
            if (u <= 0 && w[(size_t) i] > 0) return i;
        }
        return b;
    }
    std::vector<std::pair<int, float>> c;
    for (int i = 0; i < V; ++i)
        if (row[i] >= cut) c.push_back({i, row[i]});
    return mm3::sample::draw(mm3::sample::dist_c(std::move(c), {}, {}, r, hist), rng);
}

// short: a prompt's unread tokens up to this many go through the decode path, a token a step (measured: the prompt
// path's chunk costs more than that many steps)
int serve(inkling2::Model& m, const std::vector<int>& stops, int short_n) {
    for (const std::string& l : m.info_lines()) std::printf("NOTE %s\n", l.c_str());
    const inkling2::Stats& st = m.stats();
    std::printf("INFO engine=inkling2 engine_kind=inkling2 model=Inkling-2-Small ctx=%lld vram_gb=%.2f dspark=0 mtp=0 "
                "experts=%d experts_vram=%d\n", (long long) m.max_ctx(), st.vram_gb, st.experts_all, st.experts_vram);
    std::printf("READY %lld stop\n", (long long) m.max_ctx());
    std::fflush(stdout);
    std::fprintf(stderr, "maya inkling2 serve: GEN <max_new> [keys] <ids>; STOP, WRAP, QUIT\n");
    Lines lines;
    lines.start();
    std::vector<int> held;   // the tokens the context holds
    std::vector<float> lg((size_t) m.vocab());
    unsigned long long counter = 0;
    std::string err;
    auto is_stop = [&](int t) { return std::find(stops.begin(), stops.end(), t) != stops.end(); };
    for (std::string line; lines.next(line);) {
        if (line == "QUIT") break;
        if (line.empty()) continue;
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR unsupported command '%s' (GEN, STOP, WRAP and QUIT)\n", line.substr(0, 40).c_str());
            std::fflush(stdout);
            continue;
        }
        lines.stop = false;
        ++counter;
        mm3::sample::Req r;
        if (!parse_gen(line, r)) {
            std::printf("ERR expected: GEN <max_new> [keys] <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        const size_t n = r.ids.size();
        bool bad = false;
        for (int t : r.ids) bad = bad || t < 0 || t >= m.vocab();
        if (bad || (long long) n + std::max(0ll, r.max_new) > m.max_ctx()) {
            if (bad) std::printf("ERR a prompt token is outside the vocabulary (%d)\n", m.vocab());
            else
                std::printf("ERR the prompt (%zu tokens) and max_new %lld are more than the context %lld\n", n, r.max_new,
                            (long long) m.max_ctx());
            std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", n);
            std::fflush(stdout);
            continue;
        }
        // the longest prefix the context holds (one token at least is read: its logits start the answer), then the
        // latest saved position at or before it - the tokens from there are read again
        size_t lcp = 0;
        while (lcp < held.size() && lcp < n - 1 && held[lcp] == r.ids[lcp]) ++lcp;
        const size_t from = (size_t) m.rewind((int64_t) lcp);
        held.resize(from);
        const double t0 = now_s();
        auto pp = [&](size_t read) {
            const double ms = (now_s() - t0) * 1e3;
            std::printf("PP %zu %zu %.1f %.1f %zu\n", read, n, ms, ms > 0 ? 1e3 * (double) (read - from) / ms : 0.0, from);
            std::fflush(stdout);
        };
        pp(from);
        const bool want_lg = r.temperature > 0.0f || mm3::sample::penalised(r);
        bool ok = true, cancel = false;
        if (n - from <= (size_t) short_n) {
            for (size_t i = from; i < n && ok && !cancel; ++i) {
                ok = m.forward(&r.ids[i], 1, (int64_t) i, i + 1 == n && want_lg ? lg.data() : nullptr, err);
                cancel = lines.stop.load() || lines.wrap.load();
                if (ok && (i + 1 - from) % 16 == 0) pp(i + 1);
            }
            if (ok && cancel && (size_t) m.end() == n) cancel = false;   // (all read: the answer's first token may still go)
        } else {
            m.set_progress([&](int64_t done) {
                pp(from + (size_t) done);
                return !(lines.stop.load() || lines.wrap.load());
            });
            ok = m.forward(&r.ids[from], (int) (n - from), (int64_t) from, want_lg ? lg.data() : nullptr, err);
            m.set_progress(nullptr);
            cancel = ok && m.cancelled();
        }
        const double t1 = now_s();
        held.assign(r.ids.begin(), r.ids.begin() + (long) m.end());
        if (!ok) {
            std::printf("ERR %s\n", err.c_str());
            std::fprintf(stderr, "inkling2: %s\n", err.c_str());
            m.reset();
            held.clear();
        }
        if (!ok || cancel) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0\n", n, (t1 - t0) * 1e3, from);
            std::fflush(stdout);
            continue;
        }
        pp(n);
        std::mt19937_64 rng(r.seed != 0 ? r.seed : 0x9E3779B97F4A7C15ull * counter);
        const char* finish = "length";
        long long produced = 0;
        const int64_t cpu0 = st.cpu_experts, ex0 = st.experts;
        while (produced < r.max_new) {
            const int tok = want_lg ? sample(lg.data(), m.vocab(), r, held, rng) : m.top();
            std::printf("T %d\n", tok);
            ++produced;
            if (is_stop(tok)) {   // (the end token goes out too: the server's finish reason comes from it)
                finish = "stop";
                std::fflush(stdout);
                break;
            }
            if (produced % 32 == 0) {
                const double ms = (now_s() - t1) * 1e3;
                const double share = (double) (st.cpu_experts - cpu0) / (double) std::max<int64_t>(1, st.experts - ex0);
                std::printf("STAT tok_s=%.2f ms_tok=%.2f vram_hit=%.4f cpu=%.3f cpu_share=%.4f swaps=%lld\n",
                            1e3 * (double) produced / ms, ms / (double) produced, 1.0 - share, 0.0, share,
                            (long long) st.swaps);
            }
            std::fflush(stdout);
            held.push_back(tok);
            if (lines.stop.load() || lines.wrap.load()) {
                finish = "cancel";
                break;
            }
            if (produced >= r.max_new || m.end() + 1 > m.max_ctx()) break;
            if (!m.forward(&tok, 1, m.end(), want_lg ? lg.data() : nullptr, err)) {
                std::printf("ERR %s\n", err.c_str());
                std::fprintf(stderr, "inkling2: %s\n", err.c_str());
                m.reset();
                held.clear();
                finish = "cancel";
                break;
            }
        }
        held.resize((size_t) std::min<int64_t>((int64_t) held.size(), m.end()));   // (the last token given out is not read yet)
        const double t2 = now_s();
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0\n", produced, n, (t1 - t0) * 1e3, (t2 - t1) * 1e3, finish, from);
        std::fflush(stdout);
        std::fprintf(stderr,
                     "maya inkling2 serve: prompt %zu tokens = %zu reused (%zu held in common) + %zu read in %.0f ms (%.1f "
                     "tok/s), %lld generated in %.0f ms (%.1f tok/s), %s; VRAM hit %.1f%%, CPU lane %.1f%%, %lld cache moves "
                     "so far\n",
                     n, from, lcp, n - from, (t1 - t0) * 1e3, t1 > t0 ? (double) (n - from) / (t1 - t0) : 0.0, produced,
                     (t2 - t1) * 1e3, t2 > t1 ? (double) produced / (t2 - t1) : 0.0, finish,
                     100.0 - 100.0 * (double) (st.cpu_experts - cpu0) / (double) std::max<int64_t>(1, st.experts - ex0),
                     100.0 * (double) (st.cpu_experts - cpu0) / (double) std::max<int64_t>(1, st.experts - ex0),
                     (long long) st.swaps);
        if (!m.save_profile(err)) std::fprintf(stderr, "inkling2: the expert profile: %s\n", err.c_str());
    }
    return 0;
}

int usage() {
    std::fprintf(stderr,
                 "maya-inkling2 --pack <dir> (--tokens <ids> [--n-predict N] | --parity <reference dir> [--parity <dir> ...])\n"
                 "  --gpus 0,1,2,3      the GPUs (default: all)\n"
                 "  --max-context N     positions the global layers' caches hold (default 16384)\n"
                 "  --reserve-gb X      VRAM each GPU keeps free (default 1.0)\n"
                 "  --vram-experts X    the share of a layer's experts in VRAM at most (default 1.0)\n"
                 "  --layer-split a,b,c the first layer of each GPU after the first (auto: by the layers' bytes, the default)\n"
                 "  --profile F / --profile-out F   an expert profile read (the most routed in VRAM) / written\n"
                 "  --threads N         the CPU lane's workers (default: one a physical core)\n"
                 "  --log-floor N       log_scaling_n_floor in place of the pack's\n"
                 "  --layers N          (tests) the first N layers only\n"
                 "  --parity-max N      (parity) the first N positions only\n"
                 "  --parity-detail     (parity) a line a position and tensor\n"
                 "  --parity-decode     (parity) the prompt token by token (the decode path), not the prompt path\n"
                 "  --prefill-chunk N   a prompt's tokens a chunk (default 512; 1: token by token)\n"
                 "  --parity-force-picks  (parity) each reference a second time, every layer taking the reference's\n"
                 "                      picks (what is left is not the router's near-ties)\n"
                 "  --no-swap           the experts stay where the load put them (no cache moves)\n"
                 "  --tokens-file F     (tests) a prompt a line; with --profile-out: a corpus' routings\n"
                 "  --serve             the server's line protocol on stdin / stdout (docs/INKLING2_SERVING.md)\n"
                 "  --rewind N          positions a rewind may go back without reading the prompt again (default 4096)\n"
                 "  --short-prompt N    (serve) a prompt's unread tokens up to N take the decode path (default 3: measured,\n"
                 "                      the prompt path is faster from about 4 tokens on)\n"
                 "  --eos a,b           (serve) the ids that end an answer (default 200006,199999)\n"
                 "  --rewind-test DIR   (tests) a reference's prompt read, rewound and read on, against a cold read\n"
                 "  --short-bench       (tests, with --tokens) short suffixes on the prompt path and the decode path\n"
                 "  --verbose\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    inkling2::Options o;
    std::vector<std::string> parities;
    std::string tokens;
    int n_predict = 0, repeat = 1, short_n = 3;
    bool serving = false, short_bench = false;
    std::vector<int> stops = {200006, 199999};   // the model's end of sequence and <|content_model_end_sampling|>
    std::string rewind_test, tokens_file;
    std::vector<int> try_chunks;
    int64_t parity_max = -1;
    bool decode_only = false;   // (parity) the prompt token by token (the decode path) instead of the prompt path
    bool detail = false;   // (parity) a line a position and tensor
    bool force_only = false;   // (parity) that second run alone
    bool force_opt = false;   // (parity) a second run in which every layer takes the reference's picks
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--pack") o.pack = val();
        else if (a == "--gpus") o.gpus = int_list(val());
        else if (a == "--max-context" || a == "--ctx") o.max_ctx = std::atoll(val().c_str());
        else if (a == "--reserve-gb") o.reserve_gb = std::atof(val().c_str());
        else if (a == "--vram-experts") o.vram_experts = std::atof(val().c_str());
        else if (a == "--layer-split") {   // (auto: the plan's own split, by the layers' bytes)
            const std::string v = val();
            if (v != "auto") o.layer_split = int_list(v);
        }
        else if (a == "--profile") o.profile = val();
        else if (a == "--profile-out") o.profile_out = val();
        else if (a == "--threads") o.cpu_threads = std::atoi(val().c_str());
        else if (a == "--log-floor") o.log_floor = std::atof(val().c_str());
        else if (a == "--layers") o.layers = std::atoi(val().c_str());
        else if (a == "--parity") parities.push_back(val());
        else if (a == "--parity-max") parity_max = std::atoll(val().c_str());
        else if (a == "--tokens") tokens = val();
        else if (a == "--n-predict") n_predict = std::atoi(val().c_str());
        else if (a == "--repeat") repeat = std::atoi(val().c_str());
        else if (a == "--try-chunks") try_chunks = int_list(val());
        else if (a == "--verbose") o.verbose = true;
        else if (a == "--serve") serving = true;
        else if (a == "--no-swap") o.swap = false;
        else if (a == "--tokens-file") tokens_file = val();
        else if (a == "--rewind") o.rewind = std::atoi(val().c_str());
        else if (a == "--short-prompt") short_n = std::atoi(val().c_str());
        else if (a == "--eos") stops = int_list(val());
        else if (a == "--rewind-test") rewind_test = val();
        else if (a == "--short-bench") short_bench = true;
        else if (a == "--parity-detail") detail = true;
        else if (a == "--parity-decode") decode_only = true;
        else if (a == "--prefill-chunk") o.prefill_chunk = std::atoi(val().c_str());
        else if (a == "--parity-force-picks") force_opt = true;
        else if (a == "--parity-force-only") force_opt = force_only = true;
        else return usage();
    }
    if (o.pack.empty() || (parities.empty() && tokens.empty() && !serving && rewind_test.empty() && tokens_file.empty()))
        return usage();
    std::string err;
    for (const std::string& d : parities) {   // (a wrong folder: said before the load)
        Ref r;
        if (!r.load(d, err)) {
            std::fprintf(stderr, "maya-inkling2: %s\n", err.c_str());
            return 1;
        }
    }
    inkling2::Model model;
    if (!model.load(o, err)) {
        std::fprintf(stderr, "maya-inkling2: %s\n", err.c_str());
        return 1;
    }
    if (serving) return serve(model, stops, short_n);
    for (const std::string& s : model.info_lines()) std::printf("%s\n", s.c_str());
    std::printf("loaded in %.1f s, %.1f GB of VRAM\n", model.stats().load_s, model.stats().vram_gb);
    if (!tokens_file.empty()) {
        // --tokens-file F: a prompt a line (ids), each read from an empty context and --n-predict tokens decoded after
        // it - with --profile-out the routings of a corpus, the pack's default.prof
        std::ifstream tf(tokens_file);
        int np = 0;
        for (std::string line; std::getline(tf, line);) {
            std::vector<int> ids = int_list(line);
            if (ids.empty()) continue;
            if ((int64_t) ids.size() + n_predict > model.max_ctx()) ids.resize((size_t) (model.max_ctx() - n_predict));
            model.reset();
            const double t0 = now_s();
            if (!model.forward(ids.data(), (int) ids.size(), 0, nullptr, err)) {
                std::fprintf(stderr, "maya-inkling2: %s\n", err.c_str());
                return 1;
            }
            const double t1 = now_s();
            int n = 0;
            for (; n < n_predict; ++n) {
                const int tok = model.top();
                if (tok == model.eos() || !model.forward(&tok, 1, model.end(), nullptr, err)) break;
            }
            std::printf("prompt %d: %zu tokens, %.0f tok/s; %d decoded, %.1f tok/s\n", ++np, ids.size(),
                        ids.size() / (t1 - t0), n, n / std::max(1e-9, now_s() - t1));
            std::fflush(stdout);
        }
        if (!model.save_profile(err)) std::fprintf(stderr, "maya-inkling2: %s\n", err.c_str());
        return 0;
    }
    if (!rewind_test.empty()) {
        // --rewind-test <reference dir>: its prompt as A; B = A's first part and other tokens after it.  B read after
        // A (a rewind to the common prefix) against B read cold: the same next token, the last logits' cosine
        Ref rf;
        if (!rf.load(rewind_test, err)) return 1;
        const std::vector<int>& A = rf.prompt;
        std::vector<float> lw((size_t) model.vocab()), lc((size_t) model.vocab());
        int bad = 0;
        auto run = [&](const char* what, size_t keep, size_t tail, int decode, bool exact) {
            std::vector<int> B(A.begin(), A.begin() + (long) std::min(keep, A.size()));
            model.reset();
            if (!model.forward(A.data(), (int) A.size(), 0, nullptr, err)) return false;
            for (int i = 0; i < decode; ++i) {   // (B keeps the first half of the decoded tokens)
                const int t = model.top();
                if (i < decode / 2) B.push_back(t);
                if (!model.forward(&t, 1, model.end(), nullptr, err)) return false;
            }
            keep = B.size();
            for (size_t i = 0; i < tail; ++i) B.push_back(A[(i * 7 + 13) % A.size()]);
            const double t0 = now_s();
            const int64_t at = model.rewind((int64_t) keep);
            if (!model.forward(&B[(size_t) at], (int) (B.size() - (size_t) at), at, lw.data(), err)) return false;
            const double tw = now_s() - t0;
            const int topw = model.top();
            model.reset();
            const double t1 = now_s();
            if (!model.forward(B.data(), (int) B.size(), 0, lc.data(), err)) return false;
            const double tc = now_s() - t1;
            double dot = 0, a2 = 0, b2 = 0;
            for (size_t i = 0; i < lw.size(); ++i) {
                dot += (double) lw[i] * lc[i];
                a2 += (double) lw[i] * lw[i];
                b2 += (double) lc[i] * lc[i];
            }
            const double c = dot / std::sqrt(a2 * b2);
            // (exact means: the shared prefix was read by the prompt path and B resumes at one of its chunks' starts -
            //  then the same chunks follow as in the cold read; otherwise the two differ in their arithmetic's order)
            const bool ok = topw == model.top() && (c >= 0.9999 || !exact);
            bad += !ok;
            std::printf("%s: A %zu tokens (+%d decoded), B %zu tokens sharing %zu: resumed at %lld (%zu read again, %.2f s; "
                        "cold %.2f s), next token %d / cold %d, logits cosine %.6f - %s\n",
                        what, A.size(), decode, B.size(), keep, (long long) at, B.size() - (size_t) at, tw, tc, topw,
                        model.top(), c, ok ? "OK" : "DIFFERENT");
            std::fflush(stdout);
            return true;
        };
        const size_t nA = A.size();
        if (!run("diverges 500 before A's end", nA - 500, 300, 0, true) ||
            !run("diverges 37 before A's end", nA - 37, 64, 0, true) ||
            !run("diverges 2000 before A's end", nA - std::min<size_t>(2000, nA - 1), 300, 0, true) ||
            !run("diverges in the decoded part", nA, 50, 200, false) || !run("B is A and 5 more", nA, 5, 0, false)) {
            std::fprintf(stderr, "maya-inkling2: %s\n", err.c_str());
            return 1;
        }
        return bad ? 3 : 0;
    }
    if (short_bench && !tokens.empty()) {
        // --short-bench: n tokens after a held context (the --tokens prompt), through the prompt path and through the
        // decode path - where the two cross is serve's --short-prompt
        std::vector<int> ids = int_list(tokens);
        for (int n : {8, 16, 24, 32, 48, 64, 128, 256}) {
            if ((size_t) n * 2 > ids.size()) break;
            const size_t base = ids.size() - (size_t) n;
            double tp = 0, td = 0;
            for (int rep2 = 0; rep2 < 2; ++rep2) {
                model.reset();
                if (!model.forward(ids.data(), (int) base, 0, nullptr, err)) return 1;
                double t0 = now_s();
                if (!model.forward(&ids[base], n, (int64_t) base, nullptr, err)) return 1;
                tp = now_s() - t0;
                model.reset();
                if (!model.forward(ids.data(), (int) base, 0, nullptr, err)) return 1;
                t0 = now_s();
                for (int i = 0; i < n; ++i)
                    if (!model.forward(&ids[base + (size_t) i], 1, (int64_t) base + i, nullptr, err)) return 1;
                td = now_s() - t0;
            }
            std::printf("short-bench %d tokens after %zu: the prompt path %.0f ms, the decode path %.0f ms\n", n, base,
                        tp * 1e3, td * 1e3);
            std::fflush(stdout);
        }
        return 0;
    }
    std::fflush(stdout);
    std::vector<float> logits((size_t) model.vocab());

    if (parities.empty()) {
        std::vector<int> ids = int_list(tokens);
        if (ids.empty()) return usage();
        double t0 = 0, t1 = 0;
        for (int r = 0; r < std::max(1, repeat); ++r) {   // (--repeat: the prompt again from an empty context)
            model.reset();
            if (!try_chunks.empty()) {   // (--try-chunks a,b,..: run r in chunks of at most the r-th value)
                model.set_chunk(try_chunks[(size_t) r % try_chunks.size()]);
                std::printf("chunks of %d: ", try_chunks[(size_t) r % try_chunks.size()]);
            }
            t0 = now_s();
            if (!model.forward(ids.data(), (int) ids.size(), 0, nullptr, err)) {
                std::fprintf(stderr, "maya-inkling2: %s\n", err.c_str());
                return 1;
            }
            t1 = now_s();
            std::printf("prompt: %zu tokens, %.2f s, %.2f tok/s (%lld experts streamed from RAM so far)\n", ids.size(), t1 - t0,
                        ids.size() / (t1 - t0), (long long) model.stats().streamed);
            std::fflush(stdout);
        }
        const int64_t cpu0 = model.stats().cpu_experts, ex0 = model.stats().experts;
        const double cms0 = model.stats().cpu_ms;
        int n = 0;
        for (; n < n_predict; ++n) {
            const int tok = model.top();
            std::printf("%d ", tok);
            std::fflush(stdout);
            if (tok == model.eos()) break;
            if (!model.forward(&tok, 1, model.end(), nullptr, err)) {
                std::fprintf(stderr, "\nmaya-inkling2: %s\n", err.c_str());
                return 1;
            }
        }
        const double t2 = now_s();
        const inkling2::Stats& st = model.stats();
        if (n > 0)
            std::printf("\ndecode: %d tokens, %.2f tok/s; %.1f%% of the routed experts on the CPU lane (%.1f ms a token), %lld "
                        "cache moves\n",
                        n, n / (t2 - t1), 100.0 * (double) (st.cpu_experts - cpu0) / (double) std::max<int64_t>(1, st.experts - ex0),
                        (st.cpu_ms - cms0) / n, (long long) st.swaps);
        else
            std::printf("next: %d\n", model.top());
        if (!model.save_profile(err)) std::fprintf(stderr, "maya-inkling2: %s\n", err.c_str());
        return 0;
    }

    // parity: each reference as its own run from position 0 - with --parity-force-picks a second run of it in which
    // every layer takes the reference's picks
    int rc = 0;
    for (const std::string& parity : parities)
    for (int pass = force_only ? 1 : 0; pass < (force_opt ? 2 : 1); ++pass) {
    const bool force = pass == 1;
    Ref ref;
    if (!ref.load(parity, err)) return 1;
    model.reset();
    model.set_pick_hook(nullptr);
    // parity: the reference's prompt, then its greedy tokens, one a step; every tapped tensor against the dump's row
    std::vector<int> ids = ref.prompt;
    ids.insert(ids.end(), ref.greedy.begin(), ref.greedy.end());
    const int64_t np = (int64_t) ref.prompt.size();
    int64_t total = (int64_t) ids.size();
    if (parity_max > 0) total = std::min(total, parity_max);
    std::map<std::string, std::map<int, Acc>> acc;
    int64_t cur_p = 0;
    std::vector<float> rbuf, scores;
    double gap_max = 0;
    int64_t gap_big = 0;
    int64_t scores_p0 = 0;
    int scores_T = 1;
    auto step_of = [&](int64_t p) { return p < np ? 0 : (int) (p - np + 1); };
    if (force)
        model.set_pick_hook([&](int il, int* pick, int n, int64_t pos0, int T) {
            for (int t = 0; t < T; ++t) {
                const int64_t p = pos0 + t;
                const int step = step_of(p);
                if (!ref.row(step, "ffn_moe_topk-" + std::to_string(il), step == 0 ? p : 0, step == 0 ? np : 1, (size_t) n,
                             pick + (size_t) t * n))
                    return false;
            }
            return true;
        });
    // one row (a token's tensor) against the dump's
    auto one = [&](const char* name, int il, const void* data, const void* extra, size_t n, int64_t p) {
        const int step = step_of(p);
        const int64_t rows = step == 0 ? np : 1, rw = step == 0 ? p : 0;
        const std::string nm = std::string(name) + (il >= 0 ? "-" + std::to_string(il) : "");
        Acc& a = acc[name][il];
        rbuf.resize(n);
        if (std::strcmp(name, "ffn_moe_topk") == 0) {
            if (!ref.row(step, nm, rw, rows, n, rbuf.data())) return;
            const int32_t *mine = (const int32_t*) data, *theirs = (const int32_t*) rbuf.data();
            int hit = 0;
            for (size_t i = 0; i < n; ++i) hit += std::count(theirs, theirs + n, mine[i]) > 0;
            a.hit += hit;
            a.all += (int64_t) n;
            a.pos_diff += hit != (int) n;
            ++a.n;
            // how close the differing picks were: my last pick's score over my score of each reference pick I left out
            if (hit != (int) n && !scores.empty()) {
                const float* sc = scores.data() + (size_t) (p - scores_p0) * (scores.size() / (size_t) scores_T);
                float last = 1e30f;
                for (size_t i = 0; i < n; ++i) last = std::min(last, sc[mine[i]]);
                for (size_t i = 0; i < n; ++i)
                    if (std::count(mine, mine + n, theirs[i]) == 0) {
                        const double gap = (double) last - sc[theirs[i]];
                        a.rel_max = std::max(a.rel_max, gap);
                        gap_max = std::max(gap_max, gap);
                        gap_big += gap > 0.01;
                    }
            }
            if (detail && hit != (int) n) {
                std::printf("p %lld %s: mine", (long long) p, nm.c_str());
                for (size_t i = 0; i < n; ++i) std::printf(" %d(%.3f)", mine[i], ((const float*) extra)[i]);
                std::printf(" | reference");
                for (size_t i = 0; i < n; ++i) std::printf(" %d", theirs[i]);
                std::printf("\n");
            }
            return;
        }
        if (std::strncmp(name, "result_", 7) == 0) {   // (the dump has the batch's last row only)
            if (step == 0 && p != np - 1) return;
            if (!ref.row(step, nm, 0, 1, n, rbuf.data())) return;
        } else if (!ref.row(step, nm, rw, rows, n, rbuf.data())) {
            return;
        }
        const float* x = (const float*) data;
        double dot = 0, xx = 0, yy = 0, dmax = 0, ymax = 0;
        for (size_t i = 0; i < n; ++i) {
            dot += (double) x[i] * rbuf[i];
            xx += (double) x[i] * x[i];
            yy += (double) rbuf[i] * rbuf[i];
            dmax = std::max(dmax, (double) std::fabs(x[i] - rbuf[i]));
            ymax = std::max(ymax, (double) std::fabs(rbuf[i]));
        }
        const double c = dot / (std::sqrt(xx * yy) + 1e-30);
        a.cos_min = std::min(a.cos_min, c);
        a.cos_sum += c;
        a.rel_max = std::max(a.rel_max, dmax / (ymax + 1e-30));
        ++a.n;
        if (detail) std::printf("p %lld %s: cos %.6f, max rel err %.4f\n", (long long) p, nm.c_str(), c, dmax / (ymax + 1e-30));
    };
    model.set_tap([&](const char* name, int il, const void* data, size_t n, int64_t pos0, int T) {
        if (std::strcmp(name, "ffn_moe_scores") == 0) {
            scores.assign((const float*) data, (const float*) data + n);
            scores_p0 = pos0;
            scores_T = T;
            return;
        }
        const size_t row = n / (size_t) T;
        for (int t = 0; t < T; ++t)
            one(name, il, (const char*) data + (size_t) t * row * 4, (const char*) data + (n + (size_t) t * row) * 4, row,
                pos0 + t);
    });
    std::vector<int> mine;
    int greedy_same = 0, greedy_all = 0;
    const double t0 = now_s();
    for (int64_t p = 0; p < total;) {
        // the prompt in one call (the prompt path's chunks) unless --parity-decode; the greedy tokens one a step
        const int64_t n = !decode_only && p == 0 ? std::min(total, np) : 1;
        cur_p = p + n - 1;
        if (!model.forward(&ids[(size_t) p], (int) n, p, logits.data(), err)) {
            std::fprintf(stderr, "maya-inkling2: position %lld: %s\n", (long long) p, err.c_str());
            return 1;
        }
        p += n;
        if (p >= np) {
            mine.push_back(model.top());
            const size_t gi = (size_t) (p - np);
            if (gi < ref.greedy.size()) {
                ++greedy_all;
                greedy_same += ref.greedy[gi] == model.top();
            }
        }
    }
    std::printf("parity against %s: %lld positions (%lld of the prompt), %.1f s\n", parity.c_str(), (long long) total,
                (long long) std::min(total, np), now_s() - t0);
    auto get = [&](const char* name, int il) -> const Acc* {
        auto it = acc.find(name);
        if (it == acc.end()) return nullptr;
        auto jt = it->second.find(il);
        return jt == it->second.end() || jt->second.n == 0 ? nullptr : &jt->second;
    };
    std::printf("layer  l_out: cos min   cos mean   max rel err | picks same   positions differing, widest gap | attn_out   ffn_inp   "
                "ffn_out (cos min)\n");
    double worst = 2.0;
    int64_t hit = 0, all = 0;
    for (int il = 0; il < model.layers(); ++il) {
        const Acc *lo = get("l_out", il), *pk = get("ffn_moe_topk", il), *ao = get("attn_out", il),
                  *fi = get("ffn_inp", il), *fo = get("ffn_out", il);
        std::printf("%5d  ", il);
        if (lo) {
            std::printf("%.6f     %.6f   %.4f      ", lo->cos_min, lo->cos_sum / (double) lo->n, lo->rel_max);
            worst = std::min(worst, lo->cos_min);
        } else {
            std::printf("%-37s", "-");
        }
        if (pk) {
            std::printf("| %6lld/%-6lld %4lld/%-5lld %.4f ", (long long) pk->hit, (long long) pk->all, (long long) pk->pos_diff,
                        (long long) pk->n, pk->rel_max);
            hit += pk->hit;
            all += pk->all;
        } else {
            std::printf("| %-33s", "-");
        }
        auto c = [](const Acc* a) { return a ? a->cos_min : NAN; };
        std::printf("| %.6f   %.6f  %.6f\n", c(ao), c(fi), c(fo));
    }
    for (const char* name : {"Qcur", "Kcur", "Vcur", "Rcur", "kqv_out"})
        for (int il = 0; il < model.layers(); ++il)
            if (const Acc* a = get(name, il))
                std::printf("%s-%d: cos min %.6f, max rel err %.4f (%lld rows)\n", name, il, a->cos_min, a->rel_max, (long long) a->n);
    for (const char* name : {"result_norm", "result_output"})
        if (const Acc* a = get(name, -1))
            std::printf("%s: cos min %.6f, cos mean %.6f, max rel err %.4f (%lld rows)\n", name, a->cos_min,
                        a->cos_sum / (double) a->n, a->rel_max, (long long) a->n);
    std::printf("worst layer output cosine %.6f; picks %lld of %lld the same (%.3f%%)%s\n", worst, (long long) hit,
                (long long) all, all ? 100.0 * (double) hit / (double) all : 0.0,
                force ? " - every layer ran the reference's picks" : "");
    std::printf("the differing picks' selection scores: at most %.5f under my last pick's, %lld more than 0.01 under\n",
                gap_max, (long long) gap_big);
    std::printf("greedy tokens:");
    for (int tkn : mine) std::printf(" %d", tkn);
    std::printf("\nreference:    ");
    for (int tkn : ref.greedy) std::printf(" %d", tkn);
    std::printf("\ngreedy tokens the same: %d of %d\n", greedy_same, greedy_all);
    const inkling2::Stats& st = model.stats();
    std::printf("routed experts: %lld, %.1f%% on the CPU lane (%.2f ms a token)\n", (long long) st.experts,
                100.0 * (double) st.cpu_experts / (double) std::max<int64_t>(1, st.experts),
                st.cpu_ms / (double) std::max<int64_t>(1, st.tokens));
    // the spec's test ("Parity, as measured"): the same greedy tokens always; with the reference's picks every layer's
    // mean cosine >= 0.999, and on a short reference (no room for the model's own sensitivity) every row
    const bool full = model.layers() == 48 && total == (int64_t) ids.size();
    bool pass_ok = greedy_same == greedy_all;
    if (force) {
        for (int il = 0; il < model.layers(); ++il)
            if (const Acc* lo = get("l_out", il)) pass_ok = pass_ok && lo->cos_sum / (double) lo->n >= 0.999;
        if (total <= 64) pass_ok = pass_ok && worst >= 0.999;
    }
    std::printf("%s\n", !full ? "(a part of the model or of the reference: not judged)" : pass_ok ? "PARITY OK" : "PARITY FAILED");
    if (full && !pass_ok) rc = 3;
    std::fflush(stdout);
    }
    if (!model.save_profile(err)) std::fprintf(stderr, "maya-inkling2: %s\n", err.c_str());
    return rc;
}
