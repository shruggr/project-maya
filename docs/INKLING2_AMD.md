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

Measured on this PC; `--max-context 32768`, `default.prof` loads by itself, the drafter on by default.

| | without drafts (`--dspark off`) | with the DSpark drafter (default) |
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
- Dynamic VRAM/RAM expert cache: decayed counts, one mover per GPU, `--no-swap` turns it off.
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
  the window by expected tokens a millisecond; greedy and sampled acceptance; `--dspark on|off|idle`,
  `--dspark-index`, `--dspark-cum`; INFO `dspark=1`, DONE `drafts_acc` / `drafts_off`. docs/INKLING2_DSPARK.md
  ("In the engine") has the numbers.

t(T) at 2.7k context, ms a pass of T rows (`--window-test`, the drafter's taps on):

| T | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| before (commit 85ef75e) | 30.4 | 43.1 | 50.7 | 58.4 | 68.1 | 76.4 | 83.8 | 93.5 | 100.0 |
| now | 29.9 | 38.9 | 45.5 | 51.3 | 59.2 | 65.5 | 71.2 | 77.4 | 83.7 |
| now, `--dspark off` (no taps) | 29.1 | 38.8 | 45.5 | 50.8 | 59.1 | 65.9 | 71.6 | 77.5 | 83.0 |

A row more costs 6.7 ms (8.7 before). What is left of it (`INKLING2_PHASES=1`, a line a window size): the routed
experts 4.7 ms (4.3 more unique experts a layer, at the cards' memory bandwidth: the floor), the input projections'
extra columns 0.5, the router / publish / grouping 0.45, the attention 0.33 (2.2 before), the output projection 0.23,
the lane 0.2.

Two passes over the same tokens differ when the expert cache moved an expert between them (its sum on the CPU lane is
not its sum on the GPU: logits cosine 0.995-0.9999, windows or not). A drafted run moves experts at other moments
than an undrafted one (a poll a pass, not a token), so:
- with `--no-swap` the drafted greedy answer is the undrafted one token for token (`--ab --no-swap`: three prompts,
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

Engine flags: `--pack --max-context (--ctx) --layer-split --gpus --reserve-gb --vram-experts --profile (none) --profile-out
--threads --log-floor --prefill-chunk --try-chunks --repeat --rewind --short-prompt --no-swap --eos --serve --tokens
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
- **Experts**: per layer, the most-routed experts (by a profile, uint32 [48][384] routings, Hy4's `--profile`) live
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
model, ~190 GB, since the experts VRAM does not hold are copied into RAM - below that it stops, `--force-fit` goes on),
the context (up to the model's 1M, recommended by the memory left beside the model at 32 KB a token), the engine
(`build-hip/maya-inkling2`), unsloth's UD-Q2_K_XL set (`tools/inkling2/download.json`, in
`<models>/Inkling-2-Small-GGUF/UD-Q2_K_XL`), Thinking Machines' `mtp.safetensors` and `config.json` in
`<models>/Inkling-2-Small-MTP` converted by `tools/inkling2/convert_dspark.py --type q8_0`, the pack, its expert
profile (`default.prof`, copied from `data/expert-profile-inkling2.bin` where the pack has none) and
`maya-inkling2-hip.json` with `--dspark-index`. Files already there are used as they are (a pack made by hand
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
  `--max-context`, the GPUs and Drafting (`--dspark on|off` with `--dspark-index`, offered where the drafter's index is
  on this PC; maya-ik2 has no drafter); every other flag stays as the config has it.
- **The drafter's index.** The engine drafts only with the index `--dspark-index` names (the setup writes it); without
  one it decodes without drafts and says so at its start.
- **What the server adds for the model** beyond INKLING2.md's list: a tool result headed by its tool's name
  (`<|message_tool|>NAME`, by its place after the assistant's calls: the server's messages carry no call ids);
  `<|message_user|>` / `<|message_system|>` / `<|message_tool|>` as stops beside the end of sampling and
  `<|endoftext|>`; the loop guard's view of a call (`<|content_invoke_tool_json|>` .. `<|end_message|>`) and the
  prompts it continues with; a tool result's picture as a text note (a user's picture is a plain 400: text only).

The line protocol (maya-hy4's, `src/hy4/hy4_main.cpp`'s `serve`):

- Start: `NOTE <text>` / `INFO key=value ...` lines (`INFO engine=inkling2 engine_kind=inkling2 model=Inkling-2-Small
  ctx=<n> vram_gb=<x> dspark=1 logit_bias=1 ...`; `dspark=0` with `--dspark off` or no drafter index), then
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
