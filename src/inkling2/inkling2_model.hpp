// src/inkling2/inkling2_model.hpp - maya-inkling2: Thinking Machines' Inkling 2 Small (a GGUF set,
// tools/inkling2/pack_gguf.py) on the GPUs (docs/INKLING2_AMD.md).
//
// The 48 layers split over the GPUs by their memory, each holding its layers' dense matrices, its share of the routed
// experts (in VRAM, an expert's gate | up | down a blob: the most routed by a profile), the K / V caches (the
// sliding-window layers' in a ring of the window, the global layers' a row a position) and the short convolutions'
// histories.  The experts that do not fit are copied into RAM at the load and run on the CPU lane.
//
// A step takes 1 .. window_max() tokens (forward: one; window: a verify window, cut back with keep); a prompt goes
// through the layers a chunk at a time.  With Options::dspark_index the DSpark drafter (inkling2_dspark.hpp) sits on
// the last GPU and is fed every kept position's taps: draft() gives the tokens after the next one.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace inkling2 {

struct Options {
    std::string pack;
    std::vector<int> gpus;        // empty: every visible GPU
    int64_t max_ctx = 32768;      // positions the global layers' caches hold (maya-ds4's default)
    int64_t reserve_mib = 600;    // VRAM each GPU keeps free beyond the plan, MiB (maya-ds4's --reserve-mib and default)
    double vram_experts = 1.0;    // the share of each layer's routed experts kept in VRAM at most
    // the CPU lane's threads, the caller's among them (maya-ds4's meaning and default): 0 a thread a physical core;
    // n: at most n, a thread a core
    int cpu_threads = 0;
    // expert hit counts (uint32 [layers][experts]): read to place the most routed in VRAM, written back (maya-ds4's)
    std::string profile;
    bool profile_readonly = false;    // (--profile-readonly) the profile is read, never written back
    // (--profile-out) where save_profile() also writes what this run routed alone: the counts read at the load taken off
    std::string profile_out;
    std::vector<int> layer_split; // the first layer of each GPU after the first (empty: by the experts' bytes)
    int rewind = 4096;            // positions back from the furthest one read that a rewind may reach without the
                                  // prompt being read again (the sliding-window rings keep that many rows more;
                                  // MEASURE: 0.75 GB over the 40 layers against prompts read again)
    int prefill_chunk = 512;      // a prompt's tokens a forward takes through the layers at once (<= 1: token by token;
                                  // MEASURE against maya-ds4's 2048 / 4096: --try-chunks)
    int layers = 0;               // (tests) only the first `layers` layers are loaded and run (0: all)
    double log_floor = 0;         // log_scaling_n_floor in place of the pack's (0: the pack's)
    std::string dspark_index;     // the drafter's index folder (inkling2_dspark.hpp; empty: no drafter): loaded on the
                                  // last GPU, fed every position's taps by the steps and the prompt path
    float dspark_cum = 0.5f;      // its cut on the running product of the drafts' probabilities (Model::set_draft_cum)
    // VRAM the last GPU keeps for it beyond reserve_mib (fewer experts there), at a context of 32768 - what the drafter
    // reports there, measured (docs/INKLING2_DSPARK.md); a longer one adds Dspark::ctx_bytes a position.  serve/
    // models.py's IK2_DSPARK_GB is the same figure (the setup's and the Models page's)
    double dspark_reserve_gb = 3.09;
    bool expert_cache = true;     // experts move between VRAM and the CPU lane as they are used (maya-ds4's)
    bool verbose = false;
};

struct Stats {
    double load_s = 0;
    double vram_gb = 0, ram_gb = 0;          // the model's VRAM; the CPU lane's experts in RAM
    bool ram_locked = false;                 // ... page-locked (mlock took them all)
    int64_t tokens = 0;
    int experts_vram = 0, experts_all = 0;   // the routed experts in VRAM, all of them
    int64_t experts = 0, cpu_experts = 0;    // routed experts run, and how many on the CPU lane
    double cpu_ms = 0;                       // the CPU lane's time
    int64_t swaps = 0;                       // experts the cache moved into VRAM (each for a colder one moved out)
    int64_t streamed = 0;                    // experts a prompt's chunks copied from RAM to the GPUs' staging slots
    bool ram_pinned = false;                 // the RAM experts are registered with the GPUs' driver (the prompt path's)
};

class Model {
public:
    Model();
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    bool load(const Options& o, std::string& err);
    /// ids at positions pos0 .. pos0 + n - 1, one token a step (pos0 must be end(): the caches and the convolutions'
    /// histories hold the positions before it); logits_out (if given) gets the last position's logits, [vocab] F32
    /// (soft-capped, the padded rows cut).  top() is that position's argmax either way.
    bool forward(const int* ids, int n, int64_t pos0, float* logits_out, std::string& err);
    /// A verify window: T (1 .. window_max()) tokens at positions pos0 .. (pos0 = end()) in one decode step - the same
    /// arithmetic a token as forward gives it, the routed experts grouped over the window's rows.  tops [T]: each
    /// row's argmax (the token after it); logits_rows (if given): [T][vocab].  The context then ends after the T rows;
    /// keep(rows) cuts it back to the window's first `rows` (1 .. T) without reading anything again - the rows after
    /// them were drafts the target turned down.
    bool window(const int* ids, int T, int64_t pos0, int* tops, float* logits_rows, std::string& err);
    bool keep(int rows, std::string& err);
    int window_max() const;
    /// (tests) a window's rows through a token's launches, a part at a time: 1 the block types' GEMVs, 2 the experts,
    /// 4 the CPU lane, 8 the F32 GEMVs
    void set_debug(int mask);
    /// The expert cache's moves on / off from here on (Options::expert_cache).  Off, the experts stay where they are: two passes
    /// over the same tokens then give the same bits (an expert's sum on the CPU lane is not its sum on the GPU).
    void set_swap(bool on);
    /// A request's logit bias (the server's GEN logit_bias=<id>:<bias>,...; its ignore_eos bans the end tokens with
    /// -100): added on the device to every row of logits from here on - forward's last row, each window row - after the
    /// soft cap and before the argmax (top(), a window's tops) and the copy logits_out / logits_rows get.  So a greedy
    /// verdict on a draft is the biased pick, and a banned token's draft is turned down.  Ids outside the unpadded
    /// vocabulary are dropped.  None (an empty list): the logits exactly as without.  Call it between two steps.
    bool set_logit_bias(const std::vector<std::pair<int, float>>& entries, std::string& err);
    /// The drafter (Options::dspark_index) is loaded and fed
    bool dspark() const;
    /// Drafts after `anchor`, the token at end() (not read yet): up to max_n (<= 7) tokens for end() + 1 .. with the
    /// drafter's probability for each, cut where their running product falls under the cum floor.  None (true) when
    /// there is no drafter, or it does not hold the context's rows (a step with a tap or a pick hook ran).
    bool draft(int anchor, int max_n, std::vector<int>& toks, std::vector<float>& probs, std::string& err);
    void set_draft_cum(float c);
    double dspark_vram_gb() const;
    /// Back to an empty context (position 0)
    void reset();
    /// Back to a position at or before pos (<= end()): the caches keep every position and the short convolutions'
    /// histories are saved every so often (each prompt chunk's start, a prompt's end, every 64th decoded token), so
    /// the context ends at the latest saved position <= pos that the sliding-window rings still cover - returned;
    /// the caller reads the tokens from there again.  0: none was left (an empty context, as reset()).
    int64_t rewind(int64_t pos);
    /// The prompt path's progress: called after each chunk with the tokens of this forward read so far; false from
    /// it cancels the forward (which then returns true with cancelled() set and the context as it was before it).
    void set_progress(std::function<bool(int64_t)> f);
    bool cancelled() const;
    int64_t end() const;
    int top() const;
    bool save_profile(std::string& err) const;

    int vocab() const;   // the unpadded vocabulary
    int64_t max_ctx() const;
    int eos() const;
    int layers() const;
    int chunk() const;   // the prompt path's tokens a chunk
    /// (tests) chunks of at most n tokens from here on (no more than the load's Options::prefill_chunk)
    void set_chunk(int n);
    const Stats& stats() const;
    std::vector<std::string> info_lines() const;

    /// (parity, tests) a tap on the forward's tensors, named as the reference graph names them: called with the host
    /// copy of each (F32 values; "ffn_moe_topk": n int32 picks, then at data + n their n F32 weights) for layer il
    /// (-1: the head's).  Slows every step (a synchronisation a tensor).
    /// A prompt chunk comes as one call: T rows (n values in all), the first at position pos0.
    using Tap = std::function<void(const char* name, int il, const void* data, size_t n, int64_t pos0, int T)>;
    void set_tap(Tap t);
    /// (parity) each layer's picks through a hook before its experts run: ids[0 .. n) are the router's picks, the hook
    /// may write others (true: it did - their weights are then the router's for those experts).  With a tap the
    /// tap also gets "ffn_moe_scores": every expert's selection score (sigmoid + bias), before "ffn_moe_topk".
    /// (T tokens' picks, n each, the first token at position pos0)
    using PickHook = std::function<bool(int il, int* ids, int n, int64_t pos0, int T)>;
    void set_pick_hook(PickHook h);

private:
    struct Impl;
    std::unique_ptr<Impl> m;
};

}  // namespace inkling2
