# Inkling 2 Small on AMD: `maya-inkling2`

The AMD (HIP) engine for Thinking Machines' Inkling 2 Small, beside `maya-ik2`, the CUDA engine of
[INKLING2.md](INKLING2.md) (the model, the file format, the reference, the server side: read that first). It is
built the way `maya-hy4` is (src/hy4, commit 93f0486): an engine directory of its own, `src/inkling2`, that compiles
maya-ds4's ggml block-type kernels (`src/ds4/ds4_iq.cu`, `ds4_iq_cpu.cpp`, `ds4_gguf_cpu.cpp`, `MAYA_DS4_IQ_ONLY`),
from a pack of unsloth's UD-Q2_K_XL, with the model's DSpark drafter ([INKLING2_DSPARK.md](INKLING2_DSPARK.md)).
CMake builds it only with `STRATA_ENABLE_HIP`: it has never been built or run with CUDA.

To the server both engines are one family, `ik2`: the chat template (thinkingmachines' own,
`tools/ik2/chat_template.jinja`), the answer's parser (`InklingRewriter`) and the tokenizer are INKLING2.md's.
"Serving" below has what is this engine's own: its run config, its command line and the line protocol.

## Status (2026-11-20)

On v1.1.70's tree (branch `inkling2-amd`: maya-ik2's PR with this engine beside it), measured again: parity OK x4,
the rewind test's five cases OK, short decode through `--serve` 37.3 tok/s without drafts and 59.1 with them (256
tokens), `--ab` prose 33.8 -> 62.1, code 36.1 -> 59.2, thinking 37.0 -> 39.3 tok/s; through serve/server.py with the
official chat template, my_e2e's three turns 42.1 / 38.9 / 50.7 tok/s. The table below is the run before the merge
(commit f620cbc: speculative decoding wired).

Measured on this PC; `--max-context 32768`, the pack's expert profile loads by itself, the drafter on by default.
(The flags as they are named now: the drafter not loaded is `--dspark none`, maya-ds4's name; it was `off` then.)

| | without drafts (`--dspark none`) | with the DSpark drafter (default) |
|---|---|---|
| Decode, short context (12-token prompt, 512 tokens) | 37.5 tok/s | - |
| Prose: the 2.7k-token reference prompt continued, 256 tokens | 33.8 tok/s | 62.3 - 66.0 tok/s |
| Code: a chat request for a Python module, 256 tokens | 36.3 tok/s | 53.3 - 57.3 tok/s |
| Thinking: a chat question at effort 0.9, 256 tokens of thinking notes | 37.3 tok/s (39.2 with the cache warm) | 35.3 - 36.1 tok/s (39.2 with the cache warm) |
| Chat turns through serve/server.py (my_e2e: document, follow-up, code; temperature 1) | 32.0 / 33.8 / 32.1 tok/s | 38.2 / 40.5 / 45.7 tok/s |
| Prompt, 2698 tokens | ~940 tok/s | the same (every row's taps fed to the drafter) |
| Parity with llama.cpp | PARITY OK, short and long, natural and forced picks | |
| DeepSeek (maya-ds4) here, for comparison | 35.6 tok/s | 57.5 |

Done:
- Load: layer split, the expert pack, RAM experts pinned.
- Host-free decode step: device picks, a mailbox CPU lane, mapped hand-off.
- Dynamic VRAM/RAM expert cache: decayed counts, one mover per GPU, `--no-expert-cache` turns it off.
- Batched prompt path: RAM experts streamed into VRAM.
- Prefix reuse with checkpointed rewind (`--rewind`); F16 KV.
- `--serve` (maya-hy4's protocol: "Serving" below).
- **One step for T = 1 .. 9 rows** (`Impl::layer_win`): a token's step is a window of one row. Row versions of the fused
  kernels (`kv_finish_rows`, `clip2_rows`, `conv_add_rows`, `mix_rms_rows`), the attention's tiles at fixed positions
  with a sequential combine, the histories written from one buffer to another (the old one is what `keep()` cuts
  back from). A window's rows are **the token steps' logits to the bit** while the experts stay where they are
  (`--window-test`: 9 of 9 rows; `--window-bits` shows it part by part).
- The CPU lane over a window (`cpu_experts_rows`): a RAM expert several rows picked is read once for them.
- The drafter's taps: every row of every step and of every prompt chunk, F32 [rows][8][4608] on the last GPU, fed to
  `Dspark::feed_dev` once the rows are kept.
- Drafting in the decode loop and in `--serve` (`draft_round` in inkling2_main.cpp): draft -> window -> keep -> feed;
  the window by expected tokens a millisecond; greedy and sampled acceptance; `--dspark on|off|none` (maya-ds4's; `idle`, tests only),
  `--dspark-index`, `--dspark-cum`; INFO `dspark=1`, DONE `drafts_acc` / `drafts_off`. docs/INKLING2_DSPARK.md
  ("In the engine") has the numbers.

t(T) at 2.7k context, ms a pass of T rows (`--window-test`, the drafter's taps on):

| T | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| before (commit 85ef75e) | 30.4 | 43.1 | 50.7 | 58.4 | 68.1 | 76.4 | 83.8 | 93.5 | 100.0 |
| now | 29.9 | 38.9 | 45.5 | 51.3 | 59.2 | 65.5 | 71.2 | 77.4 | 83.7 |
| now, `--dspark none` (no taps) | 29.1 | 38.8 | 45.5 | 50.8 | 59.1 | 65.9 | 71.6 | 77.5 | 83.0 |

A row more costs 6.7 ms (8.7 before). What is left of it (`INKLING2_PHASES=1`, a line a window size): the routed
experts 4.7 ms (4.3 more unique experts a layer, at the cards' memory bandwidth: the floor), the input projections'
extra columns 0.5, the router / publish / grouping 0.45, the attention 0.33 (2.2 before), the output projection 0.23,
the lane 0.2.

Two passes over the same tokens differ when the expert cache moved an expert between them (its sum on the CPU lane is
not its sum on the GPU: logits cosine 0.995-0.9999, windows or not). A drafted run moves experts at other moments
than an undrafted one (a poll a pass, not a token), so:
- with `--no-expert-cache` the drafted greedy answer is the undrafted one token for token (`--ab --no-expert-cache`: three prompts,
  256 tokens each);
- with the cache on, the two part where the row's two best logits are close: prose at token 21 (gap 0.057), code at
  98 (0.059) or 111 (0.13), thinking at 7 (0.027) or 112 (0.99) - two drafted loads each, the gaps from the undrafted
  run (`INKLING2_MARGIN_LOG=1`; the median gap is 3.4 / 4.8 / 1.7, and an expert on the other side moves a logit by
  up to 1.4);
- two undrafted loads part the same way once the cache's moves fall differently: the prose prompt (the first after
  the load) gave the same 256 tokens in four loads, the code prompt after it the same 165, the thinking prompt 7.

Not done:
1. Nothing overlaps the draft (8 ms on the last GPU, the others idle) with the pass that follows it.
2. The drafter takes 3.1 GB of the last GPU (2.6 without the taps' buffers and the 32k caches): about 250 of that
   GPU's experts live in RAM instead.
3. Sampled requests copy every window row's logits to the host for the verdict (0.8 MB a row).
4. Thinking text is break-even (half of 1.5 offered drafts kept): the drafter rests there, it does not gain.

Engine flags (where maya-ds4 has the same setting, its name, unit, default and behaviour: `--max-context` 32768,
`--reserve-mib` 600 - the setup writes DeepSeek's AMD figure, 3072 on a discrete card -, `--cpu-threads` 0 (a thread a
physical core; N: at most N), `--expert-profile` (`<pack>/expert-profile.bin`, read and written back; a pack's
`default.prof` from before is read in its place), `--profile-readonly`, `--profile-out` (this run's routing alone),
`--no-expert-cache`, `--dspark on|off|none` (off: loaded, a request drafts when it says `spec=1`; none: not loaded;
`idle`, tests only, has no maya-ds4 counterpart: loaded and fed, never asked)): `--pack --max-context (--ctx)
--layer-split --gpus --reserve-mib --vram-experts --expert-profile (none) --profile-readonly --profile-out
--cpu-threads --log-floor --prefill-chunk --try-chunks --repeat --rewind --short-prompt --no-expert-cache --eos --serve --tokens
--tokens-file --n-predict --layers --dspark --dspark-index --dspark-cum --verbose`; tests `--parity [--parity-force-picks
--parity-force-only --parity-decode --parity-detail --parity-max] --rewind-test --short-bench --window-test
[--window-bits] --prompts --ab --temperature --seed --dspark-rest0`. Environment: `INKLING2_PHASES` (device phase
table, a line a window size), `INKLING2_ROUND_LOG` (a line a drafting round), `INKLING2_TIMELINE`, `INKLING2_PROF`,
`INKLING2_OLD_STEP`, `INKLING2_SWAP_BATCH`, `INKLING2_SWAP_BURST_MS`, `INKLING2_VERIFY_LOAD`, `INKLING2_EXPERT_CHECK`,
`INKLING2_NO_WARMUP`; `INKLING2_ATTN_OLD` / `INKLING2_ATTN_TK` now reach only the step with a tap or a pick hook.

## Goal and targets

- Text chat (thinking, tool calls) from unsloth's `UD-Q2_K_XL` GGUF set, served by `serve/server.py` like Hy4.
- This PC: 4x Radeon AI PRO R9700 (gfx1201, 32 GiB, PCIe 5.0 x16 each), Threadripper 9960X (24 cores, 4-channel DDR5),
  125 GB RAM, NVMe. maya-ds4 runs DeepSeek-V4.1-Flash Maya-S here at ~35 tok/s with 57% of its experts in VRAM.
- **Correctness first**: per-layer parity with llama.cpp's merged Inkling 2 implementation (the reference below).
- **Decode target: >= 25 tok/s** (stretch 35) on this PC, prompts >= 300 tok/s. Ceiling estimate: ~5.3 GB of dense
  weights + ~4.1 GB of experts read per token; layer split over 4 cards at ~600 GB/s -> ~60 tok/s if every pick were
  in VRAM.
- Not in the first version: vision (mmproj), the MTP / drafter block (its weights are not in the GGUF set: they would
  come from Thinking Machines' NVFP4 checkpoint), tensor parallel (see below).

## The files

`/mnt/data/models/llm/Inkling-2-Small-GGUF/UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-0000{1..6}-of-00006.gguf`
(shard 1 holds the metadata and tokenizer only; 1092 tensors in shards 2-6). 190.9 GiB of tensors: routed experts
185.3 GiB, everything else 5.55 GiB.

### Constants (GGUF `inkling2.*`)

| key | value |
|---|---|
| block_count | 48 (no MTP layers in the GGUF) |
| embedding_length | 4608 |
| attention.head_count / head_count_kv / key_length / value_length | 48 / 8 / 128 / 128 (GQA 6) |
| attention.layer_norm_rms_epsilon | 1e-6 |
| expert_count / expert_used_count / expert_feed_forward_length | 384 / 8 / 2304 (no shared expert) |
| expert_weights_scale | 8.0 |
| final_logit_softcapping | 30.0 |
| attention.sliding_window | 1024; pattern: layers 5, 11, 17, 23, 29, 35, 41, 47 are global, the other 40 SWA |
| d_rel (attention.relative_length) / rel_extent / rel_extent_swa | 16 / 1024 / 1024 |
| shortconv_kernel (shortconv.l_cache) | 4 |
| log_scaling_n_floor / log_scaling_alpha | **8192** / 0.1 (the GGUF's stale 128000 fixed by the pack: Open questions) |
| qk_clip_threshold / rel_clip_threshold | 22.627417 / 8.0 |
| attn_res.block_size | 6 |
| activation.situ_beta / situ_linear_beta (mlp_gate_softcap / mlp_up_softcap) | 4.0 / 25.0 |
| vocab_size / unpadded_vocab_size | 201024 / 200058 (the tokenizer list holds the 200058; rows above are padding) |
| tokenizer | `gpt2` BPE, pre `gpt-4o`; bos = eos = 200006, no BOS added; EOG also `<\|content_model_end_sampling\|>` |

### Tensors (ggml dims, ne0 first)

Global: `token_embd` (4608, 201024) Q5_K; `token_embd_norm` (4608) F32; `output_res_query` (4608) F32;
`output_norm` (4608) F32; `output` (4608, 201024) Q6_K.

Every layer `blk.L.`:

| tensor | dims | type (L0-39 / where it differs) |
|---|---|---|
| attn_norm (SWA layers only) | 4608 | F32 |
| attn_q_inp_norm, attn_k_inp_norm, attn_v_inp_norm (global only) | 4608 | F32 |
| attn_res_query (L >= 1) | (4608) SWA, (4608, 3) global | F32 |
| attn_q | (4608, 6144) | Q6_K (Q8_0 on global layers) |
| attn_gate | (4608, 6144) | Q6_K / Q8_0 |
| attn_k, attn_v | (4608, 1024) | Q8_0 |
| attn_k_conv_proj, attn_v_conv_proj (global only) | (4608, 32) | F32 |
| attn_rel | (4608, 768) | Q8_0 (48 heads x 16) |
| attn_rel_bias | (16, 1024) | F32 |
| attn_q_clip | (128, 48) | F32 (per head, per channel scale) |
| attn_rel_clip | (16, 48) | F32 |
| attn_sink_proj | (4608, 48) | BF16 |
| attn_output | (6144, 4608) | Q6_K / Q8_0 |
| attn_k_conv, attn_v_conv | (4, 1024) | F32 |
| attn_post_conv, ffn_post_conv | (4, 4608) | F32 |
| ffn_res_query | 4608 | F32 |
| ffn_norm | 4608 | F32 |
| ffn_gate_inp | (4608, 384) | F32 |
| exp_probs_b.bias | 384 | F32 (selection bias) |
| ffn_gate_up_exps | (4608, 4608, 384): rows 0..2303 gate, 2304..4607 up | IQ2_XS (L40-47: IQ3_XXS) |
| ffn_down_exps | (2304, 4608, 384) | IQ3_XXS (L40-47: IQ4_XS) |

One expert: 10.2 MB (L0-39), 13.7 MB (L40-47).

## The forward pass (one token at position p; the reference is `src/models/inkling2.cpp` in llama.cpp)

Reference checkout: `/home/shruggr/build/llama.cpp-inkling2` (master b94959a64, Inkling 2 = PR #32217 + #32429).
`rms(x)` = x / sqrt(mean(x^2) + 1e-6) with no gain; `rmsg(x, g)` = rms(x) * g.

1. `x = rmsg(token_embd[id], token_embd_norm)`. `blocks = []` (a per-token list of saved residual streams, not state
   across tokens).
2. Per layer L (`swa = L % 6 != 5`):
   - `mix(blocks, cur, Q)` (attention over depth): states S = {rms(b) for b in blocks} + {rms(cur)}; for each query
     column q of Q (1 or 3 of them): w = softmax_s(S_s . q / sqrt(4608)); out_q = sum_s w_s S_s. (The order of the
     states does not matter.)
   - Attention inputs: L = 0: in_q = in_k = in_v = cur. L >= 1: m = mix(blocks, cur, attn_res_query); SWA: all three =
     m[0]; global: in_q, in_k, in_v = m[0], m[1], m[2]. Then SWA: all three = rmsg(., attn_norm); global: rmsg with
     attn_q_inp_norm / attn_k_inp_norm / attn_v_inp_norm.
   - `Q = attn_q @ in_q` (48 x 128), `K = conv_k(attn_k @ in_k)`, `V = conv_v(attn_v @ in_v)` (8 x 128 each),
     `R = attn_rel @ in_q` (48 x 16), `t = attn_sink_proj @ in_q` (48), `g = attn_gate @ in_q` (6144).
   - Short convolution `conv(y)` with state (the last 3 inputs y of this stream, per channel), weights w (4, C):
     c[ch] = sum_{k=0..3} w[k][ch] * y_{p-3+k}[ch] (k=3 is the current input). Global layers' K and V add dynamic taps:
     d = conv_proj @ in_k (or in_v), 32 = 4 lags x 8 heads, d[lag][h]; c[h*128+i] += sum_{lag=0..3} d[lag][h] *
     y_{p-lag}[h*128+i]. Output: (y_p + c) / sqrt(n*4 + 1), n = 2 with dynamic taps, else 1. The SWA layers' K and V
     use conv without dynamic taps too (attn_k_conv / attn_v_conv exist on every layer). Four streams a layer:
     K (1024 ch), V (1024), attention post (4608), FFN post (4608) - in llama.cpp's recurrent state at channel offsets
     0, 1024, 2048, 6656.
   - Clipping: `Q_h = clip(Q_h * attn_q_clip[:, h], 22.627417)`, `K_h = clip(K_h, 22.627417)`,
     `R_h = clip(R_h * attn_rel_clip[:, h], 8)`, where clip(v, T) = v * min(1, T / (||v||_2 + 1e-6)).
     `V_h = rms(V_h)` (value norm, no gain, eps 1e-6).
   - Global layers only: Q and R times tau(p) = 1 + 0.1 * log(max(1, (p + 1) / floor)), floor = log_scaling_n_floor.
   - Scores for query head h (kv head h / 6) and key at position j, distance d = p - j:
     s = Q_h . K_j / sqrt(128) + (d < 1024 ? R_h . attn_rel_bias[:, d] : 0) - t_h; SWA layers see d < 1024 only (the
     window: masked when p - j >= 1024); global layers see every j <= p. Softmax with a zero sink logit:
     a_j = exp(s_j) / (1 + sum_j exp(s_j)). out_h = sum_j a_j V_j.
   - `o = attn_output @ (out * sigmoid(g))`, `o = conv_post_attn(o)` (no dynamic taps).
   - Residual: if L % 6 == 0: blocks.append(cur_in) and cur = o (a new stream); else cur = cur_in + o.
   - FFN: h = rmsg(mix(blocks, cur, ffn_res_query), ffn_norm). Router: logits = ffn_gate_inp @ h (F32),
     prob = sigmoid(logits); picks = top8(prob + exp_probs_b); w = prob[picks] / max(sum, 6.1035e-5) * 8.0.
     Expert e: [gate | up] = gate_up_e @ h; a = 4 tanh(gate/4) * sigmoid(gate) * 25 tanh(up/25) (SITU);
     y = sum_k w_k down_{e_k} @ a_k. `y = conv_post_ffn(y)`; cur = cur + y.
3. Head: `h = rmsg(mix(blocks, cur, output_res_query), output_norm)`, logits = output @ h, logits = 30 tanh(logits/30),
   keep the first 200058.

## State for a conversation

- KV: SWA layers: a ring of the last 1024 positions (K and V after conv / clip / norm), 40 layers x 8 heads x 128 x 2;
  global layers: every position (8 layers). Distance-based bias needs the absolute positions of the cached keys.
- Conv histories: 3 rows x (1024 + 1024 + 4608 + 4608) channels a layer.
- At 32K context: global KV 8 x 32K x 2 x 1024 x 2 B (F16) = 1 GiB; SWA 40 x 1024 x 2 x 1024 x 2 B = 160 MiB.

## The engine (design)

- `src/inkling2/`, modeled on src/hy4: `inkling2_main.cpp` (args, `--serve` protocol from hy4_main.cpp),
  `inkling2_model.cu/.hpp` (load, place, forward), `inkling2_kernels.cu/.cuh` (Inkling-only kernels: mix, convs,
  clip/norm, rel-bias attention with sinks, router), `inkling2_lane.cpp` (CPU experts), a pack reader.
- Weights from the GGUF in place through a pack index (tools/inkling2/pack_gguf.py, Hy4-style: GGUF names, stacked
  experts).
- **Layer split** over the GPUs (hidden-state hand-off between cards). Tensor parallel is out: the experts' 256-value
  blocks cannot split n_ff 2304 (maya-ds4 falls back the same way on this PC).
- **Experts**: per layer, the most-routed experts (by a profile, uint32 [48][384] routings, `--expert-profile` - Hy4's `--profile`) live
  in VRAM in ds4_iq's blob layout (gate | up | down, `tr[]` offsets, the fused GGUF gate_up maps to tr[1] = tr[0] +
  2304 * row_bytes); the rest are **copied into pinned RAM** (never paged from the GGUF on a miss - that is what made
  llama.cpp crawl here) and computed on the CPU lane (24 threads). The engine counts routings and writes a profile
  (`--profile-out`), like Hy4. Budget: ~106 GiB of experts in VRAM (57%), ~79 GiB in RAM.
- Expert math on ds4_iq's kernels with a new activation `kActSitu` (below).
- Decode first (one token a step, the prompt fed token by token), then the batched prompt path, then speed (an
  asynchronous CPU lane like maya-ds4's mailbox, a dynamic expert cache).

### The SITU activation in ds4_iq (interface between the workstreams)

`ds4_iq.cuh`: a new activation id `kActSitu` beside `kActSwiglu` / `kActOai`, with two new `Args` fields
`situ_beta` (4.0) and `situ_lb` (25.0): `a = beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb)` (lb <= 0: up unchanged).
Every gate_up form the engine uses (token, window/set, tile, the prompt paths) and the CPU lane (`iq_cpu_gate_up`
epilogue / `native_gu_rows_split`) take it. Existing activations: bit-for-bit unchanged.

## The reference (llama.cpp tensor dumps)

Tool: `tools/inkling2/ref/inkling-dump.cpp` (built as `llama-inkling-dump` in the reference checkout,
`examples/inkling-dump`), run in the ROCm container (`tools/inkling2/ref/run_dump.sh`). It evaluates the prompt as
one batch (step 0), then 3-4 greedy single-token decodes (steps 1..N), and saves every graph tensor whose name
matches a regex.

- `/mnt/data/inkling2-ref/short`: prompt `The capital of France is Paris. The capital of Germany is` (12 tokens:
  `371 151451 149879 151194 149884 152717 13 149917 151451 149879 153067 149884`), greedy continuation
  `154328 13 149917 151451` (" Berlin. The capital"). Tensors: `l_out-L`, `ffn_inp-L`, `ffn_out-L`, `attn_out-L`
  (SWA layers), `ffn_moe_topk-L` (i32 picks), `ffn_moe_probs-L`, `ffn_moe_weights*-L`, `result_norm`,
  `result_output` (last row only), and layers 0 and 5: `Qcur`, `Kcur`, `Vcur`, `Rcur`, `kq_b`, `kqv_out`.
- `/mnt/data/inkling2-ref/long`: 2698-token prompt (`tools/inkling2/ref/prompt-long.txt`), 3 steps: `l_out-L`,
  `ffn_moe_topk-L`, `attn_out-{0,5,47}`, `result_norm`, `result_output` - exercises the SWA ring and the distance cut.
- Format: `index.jsonl` (one line a tensor: step, name, file, type f32/i32, op, ne[4]); `s<step>/<name>.bin`
  contiguous, ne0 fastest; `tokens.txt` (prompt ids, then the greedy ids). Rows are tokens (ne1).
- `/mnt/data/inkling2-ref/long-f32`: the same prompt with `-fa off -ctk f32 -ctv f32` (the reference's own noise
  floor), `attn_out-L`, `ffn_inp-L`, `ffn_out-L`, `l_out-L`, `ffn_moe_topk-L` for every layer.
- **Parity, as measured.** The model is sensitive at a few positions: llama.cpp against itself (`long` vs `long-f32`,
  `tools/inkling2/ref/compare_dumps.py`) drops to a worst `l_out` cosine of 0.23 (layer 35) with 89-95% of picks the
  same in the deep layers. The jumps come at the global layers (11, 17, 29, 35, 41), while the mean stays >= 0.97.
  So a per-row minimum of 0.999 cannot be met by any implementation on long prompts. The test is instead:
  - On the short reference with the reference's picks forced (`--parity-force-picks`): every row >= 0.999.
  - On the long reference: the same greedy tokens, a per-layer **mean** cosine >= 0.999 with picks forced, and a
    natural-pass worst cosine and pick agreement no worse than `long` vs `long-f32`.
  - maya-inkling2 M1 met all of these. Short, forced: worst 0.99967. Long: natural worst 0.245 with 93.8% of picks
    the same; forced worst 0.70 with the mean >= 0.9993.

## Serving

`serve/server.py --engine maya --config <file>` runs the engine like any other. The setup makes everything, on AMD
under Linux only (`./maya.sh --setup --backend hip --gpus 0,1,2,3 --model inkling-2-small`, or Inkling 2 Small in its
model list; the Models page lists it under the other quants on an AMD install): the PC (VRAM + RAM must hold the
model, ~202 GB, `IK2_MEMORY_GB`; since the experts VRAM does not hold are copied into RAM, below that the engine stops
at its start - the setup warns and goes on, as DeepSeek's does), the context (DeepSeek's: offered from 32K up to the
model's 1M, recommended from 64K up to 256K by what VRAM + RAM hold beyond the model), the engine
(`build-hip/maya-inkling2`), unsloth's UD-Q2_K_XL set (`tools/inkling2/download.json`, in
`<models>/Inkling-2-Small-GGUF/UD-Q2_K_XL`), Thinking Machines' `mtp.safetensors` and `config.json` in
`<models>/Inkling-2-Small-MTP` converted by `tools/inkling2/convert_dspark.py --type q8_0`, the pack, its expert
profile (`expert-profile.bin`, copied from `data/expert-profile-inkling2.bin` where the pack has none) and
`maya-inkling2-hip.json` with `--dspark-index` and `--dspark none` until the speed calibration (DeepSeek's,
`tools/calibrate_ds4.py`: the config started without the drafter and with it, chat, code and tool calls) finds that
drafting pays on this PC (`--drafting measure|on|off`, `--speed-calibration`). Files already there are used as they are (a pack made by hand
included: `--repack` builds it again). `maya_quants.py`'s ids: `inkling-2-small`, `inkling-2-small-ud-q2_k_xl`. On
NVIDIA the model's engine is `maya-ik2`, which has no setup: the setup says so and stops.

By hand, the same steps:

    python tools/inkling2/pack_gguf.py --model <the GGUF set's folder>   # <model>/maya-inkling2: the pack, its tokenizer/

```json
{
 "exe": "/home/shruggr/project-maya/build-hip/maya-inkling2",
 "args": ["--pack", "/mnt/data/models/llm/Inkling-2-Small-GGUF/UD-Q2_K_XL/maya-inkling2", "--max-context", "32768",
          "--dspark-index", "/mnt/data/models/llm/Inkling-2-Small-MTP/maya-inkling2-dspark-q8_0"],
 "cwd": "/home/shruggr/project-maya",
 "tokenizer": "/mnt/data/models/llm/Inkling-2-Small-GGUF/UD-Q2_K_XL/maya-inkling2/tokenizer",
 "model_name": "inkling-2-small",
 "gpu": [0, 1, 2, 3],
 "backend": "hip",
 "lib_dirs": ["/home/shruggr/rocm10-sdk/lib/python3.12/site-packages/_rocm_sdk_devel/lib"],
 "port": 8080
}
```

- **Family.** `tools/maya_quants.py`'s `family_of` gives `ik2` by the engine's name (`maya-inkling2*`, before the
  `--pack` check that would call it DeepSeek) or by `"installer": {"family": "ik2"}`; `"inkling2"`, the family's name
  in configs written before the two engines were one family, is still read. Sampling and the thinking level are the
  family's defaults (temperature 1.0, top_p 1.0, `auto`: the template's own effort, 0.9).
- **The template.** A pack made before `pack_gguf.py` wrote the official template holds the GGUF's (the llama.cpp
  conversion's) in `tokenizer/chat_template.jinja`; the server then renders with `tools/ik2/chat_template.jinja`
  and says so at its start (`chat_template_path`; `MAYA_PACK_TEMPLATE=1` keeps the pack's).
- **The command.** `Popen([exe, "--serve", *args])` with the config's args, plus `--layer-split auto` when `gpu`
  lists more than one GPU (`engine_args`; not for `maya-ik2`, which stops at that flag), in `cwd`, with
  `HIP_VISIBLE_DEVICES` = the config's GPUs and `lib_dirs` first on the library path. The Models page changes
  `--max-context`, the GPUs and Drafting (`--dspark on|none` with `--dspark-index`, offered where the drafter's index is
  on this PC; maya-ik2 has no drafter), and for maya-inkling2 DeepSeek's rows for the settings both engines have:
  `--reserve-mib`, `--prefill-chunk`, `--cpu-threads` and `--dspark-cum` (as DeepSeek's Draft confidence,
  `--dspark-conf`); every other flag stays as the config has it.
- **The drafter's index.** The engine drafts only with the index `--dspark-index` names (the setup writes it); without
  one it decodes without drafts and says so at its start.
- **What the server adds for the model** beyond INKLING2.md's list: a tool result headed by its tool's name
  (`<|message_tool|>NAME`, by its place after the assistant's calls: the server's messages carry no call ids);
  `<|message_user|>` / `<|message_system|>` / `<|message_tool|>` as stops beside the end of sampling and
  `<|endoftext|>`; the loop guard's view of a call (`<|content_invoke_tool_json|>` .. `<|end_message|>`) and the
  prompts it continues with; a tool result's picture as a text note (a user's picture is a plain 400: text only).

The line protocol (maya-hy4's, `src/hy4/hy4_main.cpp`'s `serve`):

- Start: `NOTE <text>` / `INFO key=value ...` lines (`INFO engine=inkling2 engine_kind=inkling2 model=Inkling-2-Small
  ctx=<n> vram_gb=<x> dspark=1 logit_bias=1 ...`; `dspark=0` with `--dspark off` / `none` or no drafter index), then
  `READY <max_context> stop`. stderr is the log.
- The server sends `GEN <max_new> [key=value ...] <id,id,...>`, the whole prompt each time (the engine reuses the
  prefix it holds). Keys: `temperature` (absent or 0: greedy), `top_p`, `top_k`, `min_p`, `seed`, `penalty_*`,
  `logit_bias=<id>:<v>,...` (the server's `ignore_eos` is a bias on the stop ids); for the drafter `spec=0|1` and
  `dspark_cum=F` / `spec_cum=F`. Unknown keys are ignored. `STOP`, `WRAP`, `QUIT`. No `GENI` (text only).
- The engine answers `PP <read> <total> <ms> <tok_s> <reused>` while it reads the prompt, `T <id>` a token (a round
  with drafts gives several at once), `STAT ...` now and then (with the drafter `draft_rounds= draft_ms= verify_ms=
  drafts= draft_acc= draft_off=`, maya-ds4's names), and ends with `DONE <generated> <prompt_tokens> <prompt_ms>
  <decode_ms> <finish> <drafts_acc> <drafts_off> <reused> 0 0` (finish `stop` | `length` | `cancel`) or `ERR
  <message>`. It stops at 200006 and 199999 (`--eos`); ids past 200057, the padded rows, are never sampled.

## Build and run

```sh
R=/home/shruggr/rocm10-sdk/lib/python3.12/site-packages/_rocm_sdk_devel
cmake -S . -B build-ink-<you> -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$R -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
cmake --build build-ink-<you> --target <target> -j 24
LD_LIBRARY_PATH=$R/lib build-ink-<you>/<binary> ...
```

Each workstream uses its own build directory (`build-ink-engine`, `build-ink-kernels`); `build-ink-w1` is a
configured spare. CMake hooks: `cmake/inkling2_engine.cmake`, `cmake/inkling2_kernels.cmake` and
`cmake/inkling2_dspark.cmake`, included from CMakeLists.txt after Hy4's block inside `if(STRATA_ENABLE_HIP)`
(`_strata_gpu_*` variables in scope; tests inside `if(STRATA_BUILD_TESTS)`).

## Open questions

- `log_scaling_n_floor` (settled: **8192**). The GGUF was converted from the first config.json, which said 128000.
  Thinking Machines' 2026-11-17 commit 1df2d97d45 changed only that value, to 8192, which is also NeMo AutoModel's
  inkling2 default. pack_gguf.py reads the stale 128000 as 8192 (`STALE_FLOOR`); `--log-floor` overrides it.
  llama.cpp with this GGUF still uses 128000, so past position 8192 maya-inkling2 differs from it on purpose.
- The drafter (a later milestone, after M4). It is not in the GGUF set, but Thinking Machines ships it as
  `mtp.safetensors` (3.4 GB, BF16), downloaded to /mnt/data/models/llm/Inkling-2-Small-MTP/ with its config.json.
  - Shape (`mtp_config`): 5 dense layers (hidden 4608, 48/8 heads, head_dim 128, FFN 18432), layers 0/1/3/4 SWA 1024,
    RoPE theta 1e6, Q/K norms.
  - Block drafting, 8 tokens a block, fed by the target's hidden states at layers 5, 11, ..., 41, 48 through a fuse
    matrix and per-layer K_ctx/V_ctx projections.
  - A rank-256 GRU-style "markov" head whose correction goes through the target's own output matrix.
  - Reference: llama.cpp PR #32218 (open, on top of its DeepSeek-V4 DSpark "dflash" model). Its diff is saved at
    /mnt/data/inkling2-ref/dspark-pr32218.diff. maya-ds4's DSpark path is the template for block drafting and
    verification.
- Routing skew: on the long reference prompt the top 60% of a layer's experts take 97.5% of its picks (worst layer
  89%) - one document, so an upper bound; placement should also adapt per conversation.

## Values and their sources

Every value of this engine, its drafter, the setup, the server's and the Models page's Inkling 2 rows and the tools,
with where it comes from (2026-11-20). Sources:

- **MODEL**: the model's own files (the GGUF set, config.json, vocab.json, the official chat template); **MODEL\***:
  equal to the reference implementation (llama.cpp's `src/models/inkling2.cpp`, PR #32218's converter).
- **MAYA**: what maya-ds4, DeepSeek's setup or another Maya family does - its name, default and behaviour.
- **MEASURED**: a recorded measurement (cited). **DERIVED**: computed from the above (the formula given).
- **CHOSEN (kind)**: an internal implementation constant with the reason in its code comment - a bound it must
  cover, what it trades, a method, a name, a scope.
- **MEASURE**: a speed-affecting choice no measurement backs yet; the measurement phase measures it (against
  maya-ds4's where it has one) and the VRAM slacks become exact sizes. Unchanged until then.

### Engine

| # | Value | What it does | Source | Where it comes from |
|---|---|---|---|---|
| E1 | `max_ctx` 32768 | Positions the global layers' caches hold; the engine's context | MAYA | ds4_model.hpp `max_context = 32768`. Every recorded measurement used 32768 (INKLING2_AMD.md). The setup writes 64K or 256K instead (S15, S16). |
| E2 | `reserve_mib` 600 | VRAM each GPU keeps free beyond the plan | MAYA | ds4_model.hpp `reserve_mib = 600`. The setup writes 3072 (S27), as DeepSeek's does. |
| E3 | `vram_experts` 1.0 | The largest share of a layer's experts kept in VRAM | MAYA | ds4_model.hpp `1.0` |
| E4 | `cpu_threads` 0: a thread a physical core, the caller's among them; N at most N | CPU lane workers | MAYA | ds4_model.hpp's cpu_threads (0 and its meaning). Not mirrored: maya-ds4's lane picks between the cores, the performance cores or one NUMA node by a measurement at load; this lane is hy4::Lane |
| E5 | Default profile `<pack>/expert-profile.bin` (a pack's `default.prof` read in its place) | Expert profile read when none is given | MAYA | ds4_main.cpp's `o.pack + "/expert-profile.bin"` |
| E6 | Written back after each request and at the end; `--profile-readonly`; `--profile-out` this run's alone | Whether routing counts are saved | MAYA | ds4_model.hpp's profile_readonly, ds4_model.cu's save_profile |
| E7 | `layer_split` empty: an equal share of all bytes per GPU | How the layers are split across the GPUs | CHOSEN (design) | Tensor parallel is ruled out (256-value blocks cannot split n_ff 2304: "The engine (design)" above); `--layer-split` sets it by hand |
| E8 | `rewind` 4096 | Extra rows the sliding-window rings keep, so a rewind can reach back that far without reading the prompt again | MEASURE | Rows the SWA rings keep for a rewind: 0.75 GB over the 40 layers against prompts read again; maya-hy4 keeps 16384. Only correctness is tested (--rewind-test) |
| E9 | `prefill_chunk` 512 | Prompt tokens per prompt-path chunk | MEASURE | Against maya-ds4's auto rule (2048 on one GPU, 4096 on several): `--try-chunks` |
| E10 | `log_floor` 0, so the pack's 8192 is used | log_scaling_n_floor | MODEL | config.json `text_config.log_scaling_n_floor 8192`; inkling2.txt 8192. The GGUF still says 128000 (see T1). |
| E11 | `dspark_cum` 0.5 | Floor on the running product of the drafts' probabilities | MEASURED | DSPARK.md: the ">= 0.5" row is the best of the projected rules. Also MAYA: ds4_main.cpp `0.5f` (measured there on a Radeon 8065S, +6.5%). |
| E12 | `dspark_reserve_gb` 3.09 | VRAM the last GPU keeps for the drafter at 32K | MEASURED | The drafter reports 3.09 GB at a 32K context (INKLING2_DSPARK.md, "In the engine"); serve/models.py's IK2_DSPARK_GB, the setup's and the page's, is the same figure (tools/test_setup_ik2.py checks it). The next phase replaces it by the drafter's exact allocation |
| E13 | `expert_cache` on, `--no-expert-cache` | Expert cache on by default | MAYA | ds4_model.hpp's expert_cache, ds4_main.cpp's flag |
| E14 | `dspark_index` empty: no drafter | The drafter loads only when its index folder is named | CHOSEN (design) | The drafter is a download of its own beside the GGUF set (not in the pack, as maya-ds4's is): the config names its index, the setup writes it |
| E15 | `--short-prompt` 3 | A prompt's unread tail of at most 3 tokens goes through the decode path | MEASURE | The comment's "measured" has no recorded numbers: `--short-bench` |
| E16 | `--dspark` "on" | Drafting by default | MEASURED | INKLING2_AMD.md and DSPARK.md: prose 1.84x, code 1.47x, thinking 0.97x on a fresh load. The ds4 engine also defaults to on (ds4_model.hpp); the setup default is a separate question (S28). |
| E17 | Stops `{200006, 199999}` | Ids that end an answer | MODEL | 200006 `<\|content_model_end_sampling\|>` (the GGUF's and config.json's eos_token_id), 199999 `<\|endoftext\|>` (vocab.json); the comment says so now |
| E18 | GEN's `top_k` mm3's default, 64, unless the request sets it | No top-k by default | MAYA | ds4_main.cpp's Req (top_k 64). The server sends top_k only where the request or the config sets it (0 as 64), the same for every family |
| E19 | 40 temperature-nats cut | Tokens more than 40·T below the top are left out of sampling | MAYA | hy4_main.cpp |
| E20 | seed `0x9E3779B97F4A7C15 * counter` | Default RNG seed for each request | MAYA | hy4_main.cpp; mm3_main.cpp |
| E21 | PP every 64 tokens | How often progress lines are printed | MAYA | ds4_main.cpp's prompt windows |
| E22 | STAT every 8 tokens | How often speed lines are printed | MAYA | ds4_main.cpp's `produced % 8`; maya-hy4's too. serve/ reads the latest line |
| E23 | stdin chunk 4096 B | Read buffer | CHOSEN (bound) | A read's size only: lines are put together across reads |
| E24 | `INFO ... model=Inkling-2-Small engine=inkling2 mtp=0` | Model name and protocol keys at start | MODEL | GGUF `general.name Inkling-2-Small`. The protocol is maya-hy4's (INKLING2_AMD.md). |
| E25 | `a` = 30 ms at start | Starting estimate of a one-row pass | MEASURED | INKLING2_AMD.md (t(1) = 29.9 ms at 2.7k context) |
| E26 | `b` = 7 ms at start | Starting estimate of one more window row | MEASURED | INKLING2_AMD.md (6.7 ms a row); DSPARK.md (7 - 8 fitted) |
| E27 | `t_draft` = 10 ms at start | Starting estimate of a draft's time | MEASURE | Starting estimate; measured 8.0 - 8.5 ms (INKLING2_DSPARK.md) |
| E28 | `rest_len` 8 | Rounds without drafts after a round that did not pay | MEASURE | Only the upper side is measured (rests of 128 slept through code) |
| E29 | Rest cap 32 (doubling) | Longest rest | MEASURE | As E28 |
| E30 | 8 drafted rounds before judging | Warm-up of the gain average | MEASURE | The gain average's warm-up |
| E31 | Rest when `gain < 0.1·a` | When drafting rests | MEASURE | The comment gives the reason (thinking text: half of 1.5 drafts kept), not the threshold |
| E32 | `gain > 0.3·a` resets the rest length | Hysteresis | MEASURE | Hysteresis |
| E33 | EMA 0.9 / 0.1 | Smoothing of a, b, gain and t_draft | MEASURE | Smoothing of a, b, gain and t_draft |
| E34 | `b` floor 0.5 ms | Lower bound on the fitted row cost | MEASURE | Lower bound on the fitted row cost |
| E35 | `rest0` 0 | Rounds without a draft after k = 0 | MEASURED | main.cpp: "1 costs prose a sixth of its drafted rounds and gains thinking text nothing" |
| E36 | At most 7 drafts | Drafts per round | MODEL | config `mtp_config.block_size 8`: the anchor plus 7 drafts |
| E37 | Window sized by expected tokens per ms, on by default | How many drafts to verify | MEASURED | The rule's gain is in DSPARK.md. **Differs**: ds4's cost-aware window is off by default "until measured on GPUs" (ds4_main.cpp); ds4 uses only the fixed cum cut. |
| E38 | `ids[16]`, `tops[16]` | Window buffers | CHOSEN (bound) | At least window_max(), kWin 9 |
| E39 | +256 B per tensor | Slack per tensor in the VRAM plan | MEASURE | VRAM slack: each allocation's 256-byte alignment (al256, alloc's 256-byte minimum); becomes the exact size |
| E40 | SWA ring = `n_swa + rewind + Tc` = 1024 + 4096 + 512 = 5632 rows | Rows of each sliding-window layer's cache | DERIVED | n_swa 1024 (GGUF `attention.sliding_window`) + E8 + E9 |
| E41 | +32 MiB | Slack in the prompt path's VRAM | MEASURE | VRAM slack of the prompt path; becomes the exact size |
| E42 | `S_n()` = max(44 Mi, n_head x max_keys) floats | Prompt-path attention score scratch | DERIVED | A pass of ts tokens over nk keys takes n_head x ts x nk scores and attention_rows halves ts until it fits: one token over every key, 48 x 1M, must fit (44 Mi alone stopped a prompt at about 961K keys). The floor 44 Mi (a 512-token pass over 1,877 keys) is CHOSEN |
| E43 | `kTsMax` 512 | Tokens per pass of the prompt attention | CHOSEN (bound) | The default prompt chunk: its rel-bias table is kTsMax x 48 x 1024 floats (100 MB); a longer chunk runs passes of 512 |
| E44 | `kStage` 24 | Experts streamed per staging pass (two buffers) | CHOSEN (trade) | Two staging buffers of 24 blobs: one copied while the other's experts run - fewer passes against VRAM. Measurable (prompt tok/s) |
| E45 | +4096 B per layer | Slack per layer | MEASURE | VRAM slack a layer; becomes the exact size |
| E46 | Window scratch: 9 rows over up to 4096 keys | Attention scratch size for a window | CHOSEN (trade) | 9 = kWin rows; 4096 keys holds every SWA window (1024) and a global layer's first 4096 positions in one launch pair (58 MB a GPU); past it a window's rows run one at a time |
| E47 | +64 MiB | Decode scratch slack | MEASURE | Decode scratch slack; becomes the exact size |
| E48 | Drafter VRAM baseline at 32768 positions, + `ctx_bytes` for each position beyond | Drafter VRAM at other contexts | MEASURED | The 3.09 GB figure was measured at 32k (DSPARK.md). ctx_bytes: D10. |
| E49 | head + `8·n_embd` B | VRAM of output_norm + output_res_query (2 F32 vectors) | DERIVED | 2 x 4608 x 4 B |
| E50 | One spare expert slot per layer | Room for the cache's moves | CHOSEN (bound) | One move at a time a layer: the hot expert lands in the spare slot, the cold one's slot is the next spare |
| E51 | RAM experts + `ram_margin()` 8 GiB (INKLING2_RAM_MARGIN_GB) must fit in the RAM to be had | The load stops otherwise | MAYA | ds4_model.cu's ram_margin (8 GiB, DS4_RAM_MARGIN_GB); the setup's IK2_MEMORY_GB counts the same 8 GiB (IK2_RAM_MARGIN_GIB) |
| E52 | Staging pinned buffer of at least 256 MiB | Load-time copy buffer | MEASURE | The load's pinned staging (host memory); becomes the exact size |
| E53 | 16 blobs: the movers' bounce buffer, and the most moves per poll | Expert-cache copy buffer | CHOSEN (bound) | swap_poll's most moves a round (16): the movers' bounce holds that many blobs |
| E54 | Warm-up chunk min(Tc, 256) tokens | First-run BLAS warm-up at load | MEASURE | The first-run BLAS warm-up's chunk |
| E55 | token_embd as Q5_K / Q6_K / F32 | Embedding row types accepted | MODEL | The pack's tensors.txt: `token_embd.weight Q5_K` |
| E56 | Use counts x **0.995 per token** | How fast an expert's heat decays | MEASURE | Against maya-ds4's x0.9 every 256 tokens (DS4_CACHE_*) |
| E57 | Move when `use(hot) - 1.5·use(cold) - 0.5 > 0` | The swap rule | MEASURE | Against maya-ds4's heat > coldest x 1.25 + 1.0 |
| E58 | `swap_batch` 4 | Moves per GPU between two steps | MEASURED | model.cu: "4 and 8 decode alike, the CPU lane at 2 - 3 % of the picks" |
| E59 | `burst_s` 0 (off) | Swap burst after a prompt | MEASURED | model.cu: "the first prompt after a load was seen to take 5 s longer with it" |
| E60 | 48 rounds x 64 moves | Burst size | MEASURE | Inactive while burst_s is 0 (E59) |
| E61 | Burst only after prompts of at least 64 tokens | Burst trigger | MEASURE | Inactive while burst_s is 0 |
| E62 | Window poll: min(16, batch x min(T, 4)) moves | Moves after a window | CHOSEN (bound) | At most the bounce's 16; a window's T rows count up to 4 tokens' worth (ties to E56 / E57) |
| E63 | Prompt routings weighted min(1, 64/T) | A prompt chunk's share of the heat | MEASURE | A prompt chunk's share of the heat |
| E64 | `INKLING2_SWAP_BATCH` clamped to 1..64 | Env bounds | CHOSEN (bound) | Env bounds: at least one move; above 16 swap_poll's bounce cap takes it |
| E65 | `kSnaps` 96, `kSnapEvery` 64 | Convolution-history checkpoints: a ring of 96, one every 64 decoded tokens | MEASURE | 96 x 64 = 6144 decoded positions, more than the 4096 rewind (E8): the bound holds; the cadence is the measurable part |
| E66 | `kWin` 9 | Most rows in a window | CHOSEN (bound) | A round takes at most 8 rows (the anchor and 7 drafts, mtp_config.block_size 8); the 9th only --window-test's 9-row window uses |
| E67 | Drafter taps 2..8, the last = n_layer | Check that the drafter fits the target | MODEL | config `mtp_config.target_layer_ids` (8 entries, the last 48) |
| E68 | `kLine` 16 unsigned (64 B) | Spacing of the mapped flags | CHOSEN (bound) | 64 bytes, a cache line: each flag on its own line |
| E69 | Lane spins 20000 yields, then sleeps 200 µs | CPU lane idle policy | MEASURE | Decode tok/s against CPU use; maya-ds4's lane spins DS4_LANE_SPIN_US 20000 us |
| E70 | Depth-mix RMS eps `1e-6f` (hard-coded) | Depth-mix norm epsilon | MODEL\* | llama.cpp inkling2.cpp `ggml_rms_norm(ctx0, states, 1e-6f)`; also config `rms_norm_eps 1e-06` |
| E71 | Router weight-sum eps `1e-20f` | Router normalization | MODEL\* | llama.cpp inkling2.cpp `expert_weights_norm_eps = 1e-20f` |
| E72 | `tau = 1 + alpha·log(max(1, (p+1)/floor))` | Log attention scaling | MODEL | alpha 0.1 (GGUF / config), floor 8192 (E10) |
| E73 | `kMaxStates` 16 | Cap on depth-mix states | CHOSEN (bound) | The model needs ceil(48 / 6) + 1 = 9 states (n_layers_per_attn_res_block 6) |
| E74 | `kTile` 32, `kSeg` 128 | Token attention tiles | CHOSEN (trade) | Token attention tiles: shared memory a tile against blocks; measurable (INKLING2_PHASES) |
| E75 | `kATK` 16 | Keys per window-attention tile | MEASURED | 16-key tiles were the fastest of 64 / 32 / 16 at a 1024-key window (11.2 / 8.9 / 6.4 ms a token over the 48 layers: split_tk's comment); the window's tiles take the same size |
| E76 | `kAU` 4 for up to four rows, 8 from five | Keys per thread in window scores | MEASURED | kernels.cu (ms by T for 1 / 2 / 4 / 8 keys) |
| E77 | `kCombTiles` 8192 | Tiles per combine chunk (past 131072 keys) | CHOSEN (bound) | 8192 floats = 32 KB, a block's shared memory for the tiles' factors; past it (131072 keys at 16 a tile) the factors go a chunk at a time |
| E78 | `split_tk` 16 / 16 (`INKLING2_ATTN_TK`) | Split-attention tile keys | MEASURED | split_tk's comment: 64 keys 11.2, 32 keys 8.9, 16 keys 6.4 ms a token at a 1024-key window |
| E79 | Split scratch for at least 2048 keys, 16-key tiles | Attention scratch sizing | CHOSEN (bound) | At least 256 tiles: INKLING2_ATTN_TK's smallest tile (8) over 2048 keys |
| E80 | Clip eps `1e-6` | L2 clip denominator | MODEL\* | llama.cpp inkling2.cpp `ggml_scale_bias(norm, 1.0f, 1e-6f)` |
| E81 | Conv scale 1/sqrt(9), 1/sqrt(5) | Short-convolution output scale | MODEL\* | llama.cpp inkling2.cpp `1/sqrtf((dynamic ? 2 : 1)*n_conv + 1)`, n_conv = 4 (config `sconv_kernel_size`) |
| E82 | SITU beta 4.0, lb 25.0 | Expert activation | MODEL | GGUF `situ_beta 4.0` / `situ_linear_beta 25.0`; config `mlp_gate_softcap 4.0` / `mlp_up_softcap 25.0` |
| E83 | Fallbacks eps 1e-6, conv 4, res_block 6 | Used when the pack lacks the key | MODEL | config `rms_norm_eps`, `sconv_kernel_size`, `n_layers_per_attn_res_block` |
| E84 | No fallback: inkling2.txt must give expert_weights_scale, final_logit_softcapping, log_scaling_n_floor, bos, eos | Same | MODEL | The pack's values (8.0, 30.0, 8192, 200006); a pack without them is refused, as pack_gguf.py refuses a GGUF without them |
| E85 | Guards: n_embd % 256, head_dim ≤ 256, d_rel ≤ 256, conv == 4, n_used ≤ 16, n_expert ≤ 1024, states ≤ 16 | Geometries the engine refuses | CHOSEN (bound) | The kernels' limits: n_embd and n_ff_exp in whole 256-value blocks (ggml's k-quant super-blocks), head_dim and d_rel <= 256 (clip2 / attention_split), conv 4 (the 4-tap short convolution), n_used <= 16 and n_expert <= 1024 (hy4's router), states <= kMaxStates 16 |
| E86 | `kRowChunk` 128 | Rows per CPU work item | MAYA | hy4_lane.cpp (measured there: 64 and 32 slower on Zen 5) |
| E87 | `used[64]` | Rows tracked per CPU window | CHOSEN (bound) | At least kWin (9) rows |
| E88 | `inkling2.txt`, `tensors.txt` | Pack file names | MAYA | Hy4's pack layout (maya.py: "hy4.txt, tensors.txt, tokenizer/") |
| E89 | `--max-context --reserve-mib --vram-experts --cpu-threads --expert-profile --profile-out --prefill-chunk --eos` | Flag names | MAYA | ds4_main.cpp's flags; now also --no-expert-cache, --profile-readonly, --dspark on\|off\|none |
| E90 | `--ctx --log-floor --rewind --short-prompt --dspark-index --dspark-cum`; `--dspark idle` | Our own flag names | CHOSEN (names) | Settings maya-ds4 has no flag for (its cum floor: MAYA_DS4_DSPARK_CUM and the request's dspark_cum, the same key here); idle (tests) has no maya-ds4 counterpart |
| E91 | `INKLING2_*` env names | Test and diagnostic switches | CHOSEN (names) | The family's prefix, as DS4_* / HY4_* |
| E92 | Window-test pass: cosine ≥ 0.9995 | Test threshold | CHOSEN (bound) | Below the 9 of 9 rows found equal to the bit (above): room for a moved expert only |
| E93 | Rewind-test pass: cosine ≥ 0.9999 | Test threshold | CHOSEN (bound) | The exact cases resume at a chunk's start: the same chunks as the cold read, so only rounding is left |
| E94 | Parity: mean cosine ≥ 0.999; every row ≥ 0.999 on ≤ 64 positions | Test threshold | MEASURED | INKLING2_AMD.md (llama.cpp's own noise floor: a worst cosine of 0.23 on long prompts) |

### Drafter

| # | Value | What it does | Source | Where it comes from |
|---|---|---|---|---|
| D1 | `DsparkOptions::max_ctx` 32768 | Global-layer cache positions | MAYA | The engine's Options::max_ctx default (maya-ds4's), which the engine passes anyway |
| D2 | `rewind` 4096 (the engine passes `rewind + Tc`) | Ring margin | MEASURE | Follows E8 |
| D3 | `cum_min` 0.5 | Draft cut | MEASURED | As E11 (DSPARK.md; ds4_main.cpp) |
| D4 | `attn_tile` 8 | Keys per attention tile | MEASURED | dspark.hpp (8 / 16 / 24 keys at 300 / 2.7K / 16K) |
| D5 | `attn_gemm_min` 1024 | GEMM attention from this many keys on | MEASURED | dspark.hpp |
| D6 | `own_gemv` false | The block's GEMV kernel choice | MEASURED | dspark.hpp (within 3%) |
| D7 | `joint_index` empty: Q8_0 joint_proj | Where joint_proj comes from | MEASURED | DSPARK.md (Q8_0 = BF16 acceptance) |
| D8 | `kTaps` 8 | Taps per position | MODEL | config `mtp_config.target_layer_ids`, 8 entries |
| D9 | `kBlock` 8 | Block rows | MODEL | config `mtp_config.block_size 8` |
| D10 | `ctx_bytes` = 2·8·128·2 + 8·48·(4+2) = 6400 | VRAM per extra position | DERIVED | One global layer (`local_layer_ids [0,1,3,4]` of 5) x K, V x 8 KV heads x 128 x F16, plus block 8 x 48 heads x (F32 + F16) scores. The numbers are hard-coded, not read from dspark.txt. |
| D11 | `kFeed` 16 | Positions per feed step | MAYA | ds4_iq.cu (ncol ≤ 16) |
| D12 | `kMaxTiles` 4096 | Cap on attention tiles | CHOSEN (bound) | Past kMaxTiles tiles the tile doubles (the same sums, slower); the default GEMM form takes every context from 1,024 keys on (Sg / P16 sized from max_ctx), so the tiles see fewer. Without it (attn_gemm_min < 0) a context past the largest tile shared memory holds is refused at the load |
| D13 | `kPartAcc` 24 | Kernel accumulator bound | CHOSEN (bound) | = the 48 queries (8 rows x 6 heads a kv head) over the 2 groups of 128 threads; the load checks it |
| D14 | eps 1e-6, softcap 30 (fallbacks) | Norm eps; logit cap | MODEL | dspark.txt `eps 1e-06` (mtp_config.rms_norm_eps), `softcap 30.0` (text_config.final_logit_softcapping) |
| D15 | rope_theta fallback 1e6 | RoPE base | MODEL | mtp_config.rope_theta 1000000.0 |
| D16 | Guards NL ≤ 15, D ≤ 128, R 4..256 (multiple of 4) | Drafters it refuses | CHOSEN (bound) | The shapes the drafter's kernels were written and tested for; the model (5, 128, 256) is inside them |
| D17 | Tile clamp 4..64, else 8 | Option bounds | CHOSEN (bound) | 64: the largest tile whose shared memory fits (45 KB); 8 the measured default (D4) |
| D18 | SWA ring = n_swa + rewind + kFeed | Ring rows | DERIVED | 1024 (dspark.txt n_swa) + D2 + D11 |
| D19 | Context ring FR = min(max_ctx, rewind + 2·kFeed) | Markov context rows kept | CHOSEN (bound) | The rows a rewind can reach (D2) and two feeds |
| D20 | Markov state as int16 (x32767) | State quantization in the head | CHOSEN (method) | int16 x 32767 state in the head; acceptance matched llama.cpp's through the engine tests, no number for this scale alone |
| D21 | M rows int8, scale max/127 | Folded head matrix | CHOSEN (method) | Symmetric Q8 rows of the folded matrix |
| D22 | M built in 16-column chunks | GEMV batching | MAYA | ds4_iq.cu |
| D23 | `maya-inkling2-dspark-<kind>/`, `dspark.txt`, `format maya-inkling2-dspark 1`, `model_dir ..` | Index layout | CHOSEN (names) | tensors.txt in pack_gguf.py's format |
| D24 | The engine loads the Q8_0 copy | Drafter weight type | MEASURED | DSPARK.md |
| D25 | Taps = outputs of layers 4, 10, ..., 40 + `result_norm` | What the drafter is fed | MEASURED | DSPARK.md (the other tap variants are worse) and config `target_layer_ids` |
| D26 | RoPE NEOX (half-split) | Rotary pairing | MEASURED | DSPARK.md (adjacent pairs collapse to 0.32 / 0.69 / 0.30) |

### Setup

| # | Value | What it does | Source | Where it comes from |
|---|---|---|---|---|
| S1 | `INK_EXE` `maya-inkling2` | Engine binary | MAYA | `maya-<engine>`: DS4_EXE maya.py, HY4_EXE 804 |
| S2 | `IK2` "Inkling-2-Small" | Model id | MODEL | GGUF `general.name` / `basename` |
| S3 | `IK2_SHOWN` "Inkling 2 Small" | Name on screen | MODEL | GGUF `general.base_model.0.name` |
| S4 | `tools/inkling2/download.json` | Manifest | MAYA | DS4_MANIFEST maya.py, HY4_MANIFEST 808 |
| S5 | `IK2_PACK` "maya-inkling2" | Pack folder in the model's | MAYA | HY4_PACK "maya-hy4" maya.py |
| S6 | `IK2_QUANT` "ud-q2_k_xl" | Quant key | MODEL | unsloth's folder `UD-Q2_K_XL` (download.json; the local folder) |
| S7 | `IK2_GGUF_DIR` "Inkling-2-Small-GGUF" | GGUF folder under models | MODEL | The repo name, `unsloth/Inkling-2-Small-GGUF` (download.json) |
| S8 | `IK2_MTP_DIR` "Inkling-2-Small-MTP" | Drafter folder | CHOSEN (names) | This PC's folder name for Thinking Machines' repo |
| S9 | `IK2_DSPARK` "maya-inkling2-dspark-q8_0" | Drafter index folder | CHOSEN (names) | convert_dspark.py's Q8_0 index |
| S10 | Removed | - | - | It was not used |
| S11 | `IK2_CONFIG` "maya-inkling2-hip.json" | Config file name | MAYA | "maya-ds4-hip.json" maya.py |
| S12 | `IK2_MODEL_NAME` "inkling-2-small" | API model name | MAYA | Lower-case ids: DS4_MODEL_NAME maya.py, HY4_MODEL_NAME 848 |
| S13 | `IK2_CONTEXTS` top 1048576 | Largest context offered | MODEL | inkling2.txt `context_length`; config `model_max_length` |
| S14 | `IK2_CONTEXTS` 32K .. 1M | Contexts offered | MAYA | DS4_CONTEXTS |
| S15 | `IK2_DEFAULT_CONTEXT` 65536 | Fallback and lower bound of the recommendation | MAYA | DS4_DEFAULT_CONTEXT maya.py. It is not what the engine defaults to (E1, 32768), and nothing was measured at 64K (INKLING2_AMD.md). Measurable here. |
| S16 | `IK2_RECOMMENDED_CONTEXT` 262144 | Highest recommendation | MAYA | DS4_RECOMMENDED_CONTEXT (DeepSeek's reason: its indexer's quadratic cost). maya-inkling2 has been run end to end with a server at 256K and prompts up to 109,464 tokens (W2) |
| S17 | IK2_KV_BYTES 32768 + IK2_ATTN_BYTES 1560 on each GPU (+ IK2_DSPARK_KV_BYTES 6400 while it drafts) | Cache VRAM per context token | DERIVED | From the engine's allocations (serve/models.py says which): the global layers' K / V rows; each GPU's attention_split scratch, 48 x 130 floats a 16-key tile (past its 58 MB floor, 37,152 positions); the SWA rings are 5632 rows whatever the context. The setup and the page read the same constants |
| S18 | `IK2_DSPARK_KV_BYTES` 6400 | The drafter's per token | DERIVED | = D10 |
| S19 | `IK2_MEMORY_GB` 202 (download.json the same) | VRAM + RAM that must hold the model | DERIVED | ceil(files 190.9 GiB + the engine's RAM margin 8 GiB + the drafter's 3.09 GB = 2.9 GiB) |
| S20 | `IK2_DSPARK_GB` 3.09 | Drafter's VRAM added to the need | MEASURED | = E12, the one figure |
| S21 | `data/expert-profile-inkling2.bin` | Shipped expert profile | MEASURED | The path follows DS4_PROFILE (maya.py). The contents equal the pack's default.prof byte for byte: 15,139,200 routings = 39,425 tokens x 48 x 8. **The corpus is not recorded** (commit 8d4e2c7 says only "the profile the AMD runs used"). |
| S22 | Set up on AMD under Linux only | Platform gate | CHOSEN (scope) | maya-ik2, the NVIDIA engine, has no setup |
| S23 | VRAM + RAM < need: a warning, the setup goes on | Memory gate | MAYA | DeepSeek's check (check_pc); the warning says the engine then stops at its start |
| S24 | Recommendation pool `vram + ram - need` | Context recommendation | MAYA | ds4_recommended_context |
| S25 | 64K when RAM is unreadable or nothing fits | Fallback | MAYA | maya.py |
| S26 | Warn when a chosen context's cache does not fit | Warning | CHOSEN (warning) | A chosen context whose cache leaves the experts too little: said, on the same pool as S24 |
| S27 | `--reserve-mib` = `--vram-reserve-mib`, else hip_reserve_mib: 3072 (1024 on a Linux APU) | VRAM kept free | MAYA | DeepSeek's setup maya.py; hip_reserve_mib maya.py |
| S28 | `--dspark none` until the speed calibration; `--drafting measure \| on \| off` | Setup's drafting default | MAYA | DeepSeek's setup and tools/calibrate_ds4.py (now taking maya-inkling2's --dspark-index) |
| S29 | A setup again keeps the config's --dspark; the calibration or --drafting then sets it | Re-setup rule | MAYA | write_config_ds4's keep_args |
| S30 | Port 8080 | Default port | MAYA | maya.py |
| S31 | Kept flags `--reserve-mib --vram-experts --layer-split --cpu-threads --expert-profile --prefill-chunk --dspark-cum --rewind --short-prompt` | Hand-set flags a re-setup keeps | CHOSEN (list) | Every engine setting the setup does not ask, now with --dspark (DeepSeek's list keeps it too) |
| S32 | Sampling and effort from `MODEL_DEFAULTS["ik2"]` | Written into the config | MODEL | = T24 |
| S33 | "about a minute" | Conversion message | MEASURED | DSPARK.md (33 s niced, for both types) |
| S34 | `convert_dspark.py --type q8_0` | Only the Q8_0 copy is made | MEASURED | DSPARK.md |
| S35 | A shard 60 s newer than the pack makes it stale | Hand-made pack check | MAYA | maya.py (the same + 60) |
| S36 | Models folder `ROOT.parent/Maya-data/models` | Default location | MAYA | maya.py |
| S37 | Folders found from `--pack` (parents[1] == Inkling-2-Small-GGUF) and `--dspark-index` | Layout inference | DERIVED | From S7 / S8 / S9 |
| S38 | `IK2_PACK_NEEDS` files | Pack completeness check | MAYA | The pack layout of E88 |
| S39 | Profile copied only if its size = n_layer x n_expert x 4 (73,728) | Profile check | DERIVED | GGUF `block_count 48` x `expert_count 384` x 4 |
| S40 | Picker text: "384 experts, 8 a token", "about {IK2_MEMORY_GB} GB", "up to a 1M context" | Model list row | MODEL | GGUF `expert_count` / `expert_used_count` / `context_length`; the 190 is S19 |
| S41 | The setup's drafting message is DeepSeek's (no figure) | Drafting message | - | The figure is IK2_DSPARK_GB where shown |

### Server and Models page

| # | Value | What it does | Source | Where it comes from |
|---|---|---|---|---|
| W1 | 32K .. 1M | Contexts on the Models page | MAYA | = S14 (models.py's DS4_CONTEXTS) |
| W2 | `CONTEXT_TESTED["inkling2"]` 131072 | Where the page starts saying "untested" | MEASURED | The v1.2 benchmark: a server at --max-context 262144, prompts up to 109,464 tokens; the measurement phase raises it |
| W3 | `IK2_KEYS` + reserve_mib, prefill_chunk, cpu_threads, dspark_cum (maya-inkling2 only) | Settings the page edits | MAYA | DS4_KEYS; dspark_cum is dspark_conf's counterpart, in DeepSeek's Draft confidence row |
| W4 | VRAM kept free (--reserve-mib) for maya-inkling2 | - | MAYA | DeepSeek's row |
| W5 | `--dspark` read as "on" when absent; off and none read as Off | Current Drafting state | MAYA | The engine's default, as maya-ds4's; the page writes none (DeepSeek's) |
| W6 | Context fallback 32768 | Context when `--max-context` is absent | MAYA | = E1 (ds4_model.hpp) |
| W7 | Trained context from inkling2.txt `context_length` | Slider top | MODEL | inkling2.txt `context_length 1048576` |
| W8 | Drafter searched at `<models>/Inkling-2-Small-MTP/maya-inkling2-dspark-q8_0` and `pack.parents[2]/...` | Finds the index | DERIVED | From S7 / S8 / S9 |
| W9 | The page's cost: the server's kv_bytes + kv_gpu_bytes x GPUs + dspark_kv_bytes while it drafts | Cache GB shown for a context | DERIVED | = S17, from serve/models.py |
| W10 | "About {IK2_DSPARK_GB} GB" (3.09) | Drafting hint | MEASURED | = E12 |
| W11 | Levels none, auto, low, medium, high, xhigh, max | Thinking levels offered | MODEL | The template's levels as the other PR's server takes them (frontend.py's INKLING_EFFORT): its effort_kwargs reads minimal as thinking off (0), so the page does not offer it beside Off |
| W12 | Default "auto" (the template's 0.9) | Default thinking level | MODEL | chat_template.jinja:5 |
| W13 | 0 / 0.2 / 0.7 / 0.9 / 0.99 in the titles | Labels | MODEL | chat_template.jinja:1 |
| W14 | "Inkling 2 Small (maya-inkling2)", NAME_WORDS `inkling` | Labels | MODEL | GGUF base_model name |
| W15 | `INKLING_TEXT_ONLY` "Inkling-2-Small" | Name in picture errors | MODEL | GGUF `general.name` |
| W16 | Stops on `<|message_user|>`, `<|message_system|>`, `<|message_tool|>` | Turn-end guard | MODEL | chat_template.jinja:2 role tokens; vocab ids 200000 / 200002 / 200003. The template ends a model turn with `<\|content_model_end_sampling\|>` (124). Treating the role tokens as stops is our addition. |
| W17 | Call end `<|end_message|>` | Tool-call parsing | MODEL | chat_template.jinja:119 |
| W18 | GGUF-template detection: has `<|content_invoke_tool_json|>` and `<|message_model|>`, no `effort_map`; `MAYA_PACK_TEMPLATE=1` | Uses the official template | MODEL | Markers: chat_template.jinja:1, 112. The env name is CHOSEN. |
| W19 | maya-inkling2 gets `--layer-split auto` (maya-ik2 does not) | Engine command | MAYA | The server's rule for any multi-GPU engine (server.py) |
| W20 | Call open `<|content_invoke_tool_(json\|text)|>`, close `<|end_message|>` | Loop guard | MODEL | chat_template.jinja:112, 119; vocab 200057 (`_text`) |
| W21 | A tool result is named by its position (the k-th result answers the k-th call) | Tool result header | CHOSEN (other PR) | frontend.py's inkling_messages (PR #1079's): OpenAI's and Anthropic's order |
| W22 | `IK2_OTHER` ("inkling-2-small-ud-q2_k_xl", "unsloth") | Download row | MODEL | download.json (unsloth) |
| W23 | Offered only on `backend == "hip"`; size = set + drafter | Download offer | CHOSEN (scope) | = S22; the size is download.json's sums |

### Tools

| # | Value | What it does | Source | Where it comes from |
|---|---|---|---|---|
| T1 | `STALE_FLOOR {128000: 8192}` | Fixes the GGUF's floor | MODEL | config.json `log_scaling_n_floor 8192` against GGUF `log_scaling_n_floor` / `attention.log_scale_floor` 128000. The Thinking Machines commit hash (pack_gguf.py; INKLING2_AMD.md) cannot be checked locally; config.json can. |
| T2 | Output `<model>/maya-inkling2` | Pack folder | MAYA | = S5 |
| T3 | No fallback: the GGUF must give both keys | Used when the GGUF lacks the key | MODEL | pack_gguf.py's other required keys |
| T4 | `attention.log_scale_floor` read first, `log_scaling_n_floor` second | Key order | MODEL | Both are 128000 in this GGUF |
| T5 | Geometry guards | Refuses other shapes | CHOSEN (bound) | = E85 |
| T6 | Chat template: tools/ik2/chat_template.jinja over the GGUF's | Pack tokenizer | MODEL | The official template |
| T7 | `pre gpt-4o`, `add_bos False`, bos / eos 200006 | tokenizer.json | MODEL | GGUF `tokenizer.ggml.pre`, `add_bos_token`, `bos/eos_token_id` |
| T8 | `ARCH "dflash"` | Drafter GGUF architecture | MODEL\* | PR #32218's converter (DSPARK.md) |
| T9 | `N_TENSORS` 84 | Tensor count check | MODEL | mtp.safetensors has 84 BF16 tensors (DSPARK.md; expect_shapes builds them from mtp_config) |
| T10 | `block_size == 8` required | Config check | MODEL\* | "the PR's own checks"; config block_size 8 |
| T11 | `sample_from_anchor false`, `has_confidence_head false`, `attention.causal false` | GGUF keys | MODEL\* | PR converter (DSPARK.md) |
| T12 | eos fallback 200006 | Tokenizer key | MODEL | config `eos_token_id 200006` |
| T13 | eot = eos; `mask_token_id 0` | Tokenizer keys | MODEL\* | DSPARK.md (the PR's) |
| T14 | 1-D tensors and fuse_weight as F32 | Tensor types | MODEL\* | The PR forces the fuse (comment) |
| T15 | `--type` default "both" | Which copies are made | CHOSEN | Both copies for the tests; the setup passes q8_0 (S34) |
| T16 | Index names and format | Index layout | CHOSEN (names) | = D23 |
| T17 | Verify bound: \|d\|·(0.5 + 127·2^-11)·1.001 + 127·2^-25·1.001 | Q8_0 read-back tolerance | DERIVED | The x1.001 is CHOSEN slack over the derived bound |
| T18 | name "Inkling-2-Small DSpark", base_model URL | GGUF metadata | MODEL | download.json repo |
| T19 | unsloth repo, pinned revision, 6 shard sizes and sha256 | Model download | MODEL | The sizes match the local shards byte for byte (sha256 not recomputed) |
| T20 | 202 | Memory need | DERIVED | = S19 |
| T21 | Drafter repo, revision, mtp.safetensors 3,423,429,800 B, config.json 2,767 B | Drafter download | MODEL | Match the local files |
| T22 | Family `ik2`, MODEL / SHOWN names | Family table | MODEL | = S2 / S3 |
| T23 | OTHERS `ud-q4_k_xl`, `ud-q3_k_xl`, `ud-q2_k_xl` | Quant labels | MODEL / CHOSEN | UD-Q2_K_XL is on disk; Q4 / Q3 are maya-ik2's (the other PR's) labels |
| T24 | `MODEL_DEFAULTS["ik2"]` temperature 1.0, top_p 1.0, "auto" | Family sampling | CHOSEN (other PR) | maya_quants.py's MODEL_DEFAULTS (blame 1ef2ea1): "the card and files name no sampling" |
| T25 | `IK2_IDS` aliases | Setup ids | CHOSEN (names) | Setup ids |
| T26 | `IK2_FAMILY_NAMES`, `IK2_EXES`, the `UD-...` regex | Family detection | CHOSEN (names) | Family detection; the engine names are MAYA (S1) |

The measurement phase's list (MEASURE above): the prompt chunk (E9, against maya-ds4's 2048 / 4096); the rewind
reach (E8, D2); the short prompt (E15); the drafting rounds' scheduler (E27 - E34); the expert cache's decay and swap
rule (E56, E57, against maya-ds4's x0.9 every 256 tokens and heat > coldest x 1.25 + 1.0); the burst and the warm-up
(E54, E60, E61, E63, E65); the lane's spin (E69); the VRAM slacks (E39, E41, E45, E47, E52), which become exact sizes.
