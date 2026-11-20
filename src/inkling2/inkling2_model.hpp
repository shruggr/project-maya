// src/inkling2/inkling2_model.hpp - maya-inkling2: Thinking Machines' Inkling 2 Small (a GGUF set,
// tools/inkling2/pack_gguf.py) on the GPUs (docs/INKLING2_PORT.md).
//
// The 48 layers split over the GPUs by their memory, each holding its layers' dense matrices, its share of the routed
// experts (in VRAM, an expert's gate | up | down a blob: the most routed by a profile), the K / V caches (the
// sliding-window layers' in a ring of the window, the global layers' a row a position) and the short convolutions'
// histories.  The experts that do not fit are copied into RAM at the load and run on the CPU lane.
//
// This version runs the decode path only: one token a step through every layer (a prompt is fed token by token).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace inkling2 {

struct Options {
    std::string pack;
    std::vector<int> gpus;        // empty: every visible GPU
    int64_t max_ctx = 16384;      // positions the global layers' caches hold
    double reserve_gb = 1.0;      // VRAM each GPU keeps free beyond the plan
    double vram_experts = 1.0;    // the share of each layer's routed experts kept in VRAM at most
    int cpu_threads = -1;         // the CPU lane's workers besides the caller (-1: one a physical core but the first)
    std::string profile;          // an expert profile (uint32 [layers][experts] routings): the most routed in VRAM
    std::string profile_out;      // where save_profile writes it (the read one's counts with this run's)
    std::vector<int> layer_split; // the first layer of each GPU after the first (empty: by the experts' bytes)
    int rewind = 4096;            // positions back from the furthest one read that a rewind may reach without the
                                  // prompt being read again (the sliding-window rings keep that many rows more)
    int prefill_chunk = 512;      // a prompt's tokens a forward takes through the layers at once (<= 1: token by token)
    int layers = 0;               // (tests) only the first `layers` layers are loaded and run (0: all)
    double log_floor = 0;         // log_scaling_n_floor in place of the pack's (0: the pack's)
    bool swap = true;             // the expert cache: the experts routed to lately move into VRAM, colder ones out
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
