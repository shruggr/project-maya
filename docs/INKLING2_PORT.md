# Inkling 2 Small on Maya: the port spec (maya-inkling2)

Working spec for bringing Thinking Machines' Inkling 2 Small into Maya as its own engine, `maya-inkling2`,
built the way `maya-hy4` is (src/hy4, commit 93f0486): an engine directory of its own that compiles maya-ds4's
ggml block-type kernels (`src/ds4/ds4_iq.cu`, `ds4_iq_cpu.cpp`, `ds4_gguf_cpu.cpp`, `MAYA_DS4_IQ_ONLY`).

Branch: `inkling2-engine`.

## Status (2026-11-20, commit 85ef75e)

Measured on this PC; `--max-context 32768`, `default.prof` loads by itself.

| | |
|---|---|
| Decode, short context | 36.9 tok/s |
| Decode after a 2.7k-token prompt | 32.8 tok/s |
| Decode at 16k / 32k context | 26.0 / 22.0 tok/s |
| Chat turns through serve/server.py (my_e2e: document, follow-up, code) | 31.7 / 32.7 / 32.0 tok/s |
| Prompt, 2698 tokens | ~945 tok/s (~1190 at 16k) |
| Parity with llama.cpp | PARITY OK, short and long, natural and forced picks |
| DeepSeek (maya-ds4) here, for comparison | 35.6 tok/s without drafting, 57.5 with DSpark |

Done:
- Load: layer split, the expert pack, RAM experts pinned.
- Host-free decode step: device picks, a mailbox CPU lane, mapped hand-off.
- Dynamic VRAM/RAM expert cache: decayed counts, one mover per GPU, `--no-swap` turns it off.
- Batched prompt path: RAM experts streamed into VRAM.
- Prefix reuse with checkpointed rewind (`--rewind`); F16 KV.
- `--serve` (maya-hy4's protocol, docs/INKLING2_SERVING.md).
- Split-K decode attention.
- Verify window: `Model::window` / `keep`, T ≤ 9 tokens.
- The DSpark drafter as a standalone module (src/inkling2/inkling2_dspark.*; docs/INKLING2_DSPARK.md).

Not done (drafting is not wired yet):
1. One code path for T = 1 and windows. Window rows match the token step only to cosine 0.995-0.9999 (same tokens),
   because the window uses unfused row kernels.
2. The CPU lane over a window: a RAM expert picked by several rows is read once a row today.
3. The drafter's taps out of the window step and the prompt path (all rows) to the drafter's GPU.
4. Drafting in the decode and serve loops: the k policy from t(T), sampled acceptance, and `--dspark` flags.
5. End-to-end drafted measurement.

t(T) at 3k context, ms per window pass, for T = 1..9: 30.4, 43.1, 50.7, 58.4, 68.1, 76.4, 83.8, 93.5, 100.0.
Unique experts grow by ~4.3 per layer per row (~4.3 ms at bandwidth); the rest of the ~8 ms per row is unmeasured
(there are no phase marks in the window step yet).

Engine flags: `--pack --max-context (--ctx) --layer-split --gpus --reserve-gb --vram-experts --profile (none) --profile-out
--threads --log-floor --prefill-chunk --try-chunks --repeat --rewind --short-prompt --no-swap --eos --serve --tokens
--tokens-file --n-predict --layers --verbose`; tests `--parity [--parity-force-picks --parity-force-only --parity-decode
--parity-detail --parity-max] --rewind-test --short-bench --window-test`. Environment: `INKLING2_PHASES` (device phase
table, token step), `INKLING2_TIMELINE`, `INKLING2_PROF`, `INKLING2_OLD_STEP`, `INKLING2_SWAP_BATCH`,
`INKLING2_SWAP_BURST_MS`, `INKLING2_ATTN_OLD`, `INKLING2_ATTN_TK`, `INKLING2_VERIFY_LOAD`, `INKLING2_EXPERT_CHECK`,
`INKLING2_NO_WARMUP`.

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
configured spare. CMake hooks: `cmake/inkling2_engine.cmake` and `cmake/inkling2_kernels.cmake`, included from
CMakeLists.txt next to Hy4's block (inside `if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)`, so `_strata_gpu_*`
variables are in scope; tests inside `if(STRATA_BUILD_TESTS)`).

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
