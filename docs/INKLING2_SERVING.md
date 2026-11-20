# Inkling 2 Small: serving notes (the integration workstream)

How `serve/server.py` serves Thinking Machines' Inkling 2 Small through `maya-inkling2` (the engine of
docs/INKLING2_PORT.md): the tokenizer, the chat template, the answer's parsing, the config, and the line protocol the
server expects from the engine. Everything here runs without the engine; the tests are
`tools/test_inkling2_tokenizer.py` and `serve/test_inkling2_frontend.py`.

## Tokenizer (tools/strata_tokenizer.py)

- The GGUF says `tokenizer.ggml.model gpt2`, `pre gpt-4o` (llama.cpp's converter now writes `inkling2` for the same
  tokenizer; both are `LLAMA_VOCAB_PRE_TYPE_GPT4O`), 200,058 tokens (the model's vocab of 201,024 is padded; rows past
  200,057 are never produced), 424,413 merges (no duplicates, every merge's result in the vocabulary), 60 control
  tokens (type 3) from 199998 up, `bos = eos = 200006` (`<|content_model_end_sampling|>`), no BOS added. llama.cpp's
  end-of-generation set: 200006 and 199999 (`<|endoftext|>`).
- `Tokenizer(pre="gpt-4o" | "inkling2")` splits **as llama.cpp does**: GPT-4o's regex run over the text collapsed by
  Unicode category (`unicode_regex_split`'s std::regex path; Unicode 15.1 tables, so characters new in 16.0 count as
  undefined; std::regex's ASCII `\s`). That differs from the original regex (tokenizers / tiktoken) for cased non-ASCII
  letters: `ÇA` is `Ç` + `A` there and one piece in the original. A pack's `tokenizer.json` `"pre_regex": "unicode"`
  switches to the original regex. Default: `llama` (the port's reference).
- **Parity with llama.cpp** (checkout b94959a64, `llama-tokenize --ids --no-escape`, built in the ROCm container, run
  CPU-only): 938 strings x {special tokens parsed, not parsed} = 1,876 tokenizations, 399,503 tokens - **all
  identical** (unicode of many scripts, combining marks, emoji, Unicode 16 letters, control and separator bytes, code,
  numbers, contractions, whitespace runs, special tokens and their near misses, rendered chat prompts, the repo's docs
  and sources, 800 fuzz strings), plus 40 fresh random strings through the live oracle test. Both reference prompts
  match: `prompt-short.txt` -> the 12 ids of `/mnt/data/inkling2-ref/short/tokens.txt`, `prompt-long.txt` -> its 2,698.
  With the original regex instead, 2 of the 1,876 differ (`Ça va? ÇA VA!`).
  198 of the strings and their ids are kept in `serve/fixtures/inkling2/tokenizer_golden.json`.
- **Aliases** (what the server reads for an output token): Inkling's structural tokens - `<|message_model|>`,
  `<|content_thinking|>`, `<|content_text|>`, `<|content_invoke_tool_json|>`, `<|content_invoke_tool_text|>`,
  `<|end_message|>`, `<|content_tool_error|>` - decode to one private-use character each (U+10FFF0..U+10FFF6,
  `strata_tokenizer.INKLING2_MARKS` = `frontend.INKLING_MARKS`). The model writing such a token's *text* out of
  ordinary pieces (quoting its own format) stays text. Applied automatically for a gpt-4o vocabulary that has Inkling's
  tokens, so a pack's `tokenizer.json` needs no `aliases` key; prompts still encode the tokens by their text.
- **Where the files are**: the pack's `tokenizer/` folder, `<model>/maya-inkling2/tokenizer/`
  (tools/inkling2/pack_gguf.py writes it: vocab.json, merges.txt, token_type.json, tokenizer.json `{pre, special_ids,
  add_bos}`, chat_template.jinja - the server reads that as it is). `strata_tokenizer.tokenizer_files(kv)` returns the
  same folder from a GGUF's metadata (any reader's kv) with the shipped pattern, `pre_regex` and the aliases written
  out; the pack tool may call it instead of its own. An export of it is at
  `/mnt/data/models/llm/Inkling-2-Small-GGUF/UD-Q2_K_XL/maya-inkling2-tokenizer/` (identical vocabulary and merges).

## Chat template (serve/frontend.py, `ChatTemplate.inkling`)

- Detected by its tokens (`<|message_model|>`, `<|content_invoke_tool_json|>`, `<|content_model_end_sampling|>`); the
  template is the GGUF's own (`serve/fixtures/inkling2/chat_template.jinja`, the pack's copy at run time), rendered
  with jinja2 as transformers does. Checked against llama.cpp's own chat code (`common_chat_templates_apply`,
  18 conversations: system / developer, history with reasoning, tool declarations, single and parallel calls, nested
  JSON arguments with floats and unicode keys, every effort level): identical, except an assistant turn with nothing
  in it, which the server never renders (its policy for every model).
- No generation prefix: the prompt ends with the last message's `<|end_message|>`; the model opens its own turn with
  `<|message_model|>`.
- Thinking levels by the template's own names: none 0.0, minimal 0.1, low 0.2, medium 0.7, high 0.9 (default),
  xhigh / max 0.99. Clients: OpenAI `reasoning_effort` / `reasoning.effort` / `chat_template_kwargs.reasoning_effort`
  (all seven words, `auto` = the template's default), `enable_thinking: false` = none; Anthropic `thinking.disabled` =
  none, `budget_tokens` < 2K low, < 8K high, else max, `output_config.effort`. Numbers (`0.5`) are not taken (the
  server's effort words only). The page offers none / low / medium / high / xhigh / max.
- Tool results are headed by their tool's name, as the official template wants (`message.name` or the call their
  `tool_call_id` answers): the server's messages carry neither, so `inkling_messages` names each result by its place
  after the assistant turn's calls (OpenAI's and Anthropic's order).
- A message of several text parts (Anthropic's text blocks, OpenAI's content arrays) stays a list of parts - a block
  each, as llama.cpp renders them (other models still get them joined).
- Pictures: text only (the engine has no vision tower): a user's picture is a plain 400 ("Inkling 2 Small on
  maya-inkling2 reads text only"), a tool result's picture becomes a short text note.
- A required call (`tool_choice` a named function): the prompt ends with
  `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":` and thinking off; `required` without a name:
  `<|message_model|>` alone (the model names the tool - or writes another block).

## The answer (serve/frontend.py, `InklingRewriter`, `OutputParser(inkling=True)`)

- Blocks: `H<|content_thinking|>...E` is reasoning, `H<|content_text|>...E` the answer, `H NAME<|content_invoke_tool_json|>
  {"name":"NAME","args":{...}}E` a call (H = `<|message_model|>`, E = `<|end_message|>`), the turn's end
  `<|content_model_end_sampling|>` (a stop token, never text). Consecutive thinking blocks concatenate (as llama.cpp's
  parser does). No text is searched for a tag: `</think>`, `<tool_call>` in the model's words are text.
- A call's arguments stream: the tool's name as soon as its block opens (`tool_start`), a string argument as it is
  written, other values whole at their end. Another JSON form (keys in another order, spaces;
  `<|content_invoke_tool_text|>`) is read whole at the block's end; a call that is not JSON is the answer's text (and
  logged as malformed); one cut off by max_tokens ends as other models' do (incomplete / its JSON closed).
- Text outside any block is answer text (a model that writes no blocks still answers); whitespace between blocks is
  dropped.
- Checked against llama.cpp's Inkling parser (common/parsers/inkling.cpp via `common_chat_session`): the 10 answers of
  its own tests (thinking, content, cut off in both, single / parallel calls, a block closer inside a string argument,
  an argument-less call, two thinking blocks) - identical at every delta size.

## Server wiring (serve/server.py, serve/loop_guard.py)

- Stop ids: `<|content_model_end_sampling|>` (= the tokenizer's eos, the engine's own end), `<|endoftext|>`, and
  `<|message_user|>` / `<|message_system|>` / `<|message_tool|>` (another role's block: the turn is over).
  `<|end_message|>` is not a stop (it ends every block).
- No end-of-thinking token (`<|end_message|>` ends every block): `think_end_id` is None, so no thinking budget (the
  dashboard says "not supported") - a budget would need the engine to close a thinking block with three tokens.
- The loop guard knows a call by `<|content_invoke_tool_json|>` ... `<|end_message|>` (their marks); its "keep going"
  prompt starts after the model's own `<|end_message|>`; a blocked call is answered after the call's block.
- Responses / Anthropic / OpenAI / the Ollama-style API all go through the same conversion (`inkling=` flag).
- The web page: family `inkling2` (by the model name "inkling..."), its six levels and their titles, its own saved
  chat settings, the Models page's fields (context, GPUs; images and parallel answers "not yet").

## The config, and what the server runs

Family `inkling2` (tools/maya_quants.py: `family_of` by `installer.family` or the exe's name `maya-inkling2*` - before
the `--pack` check that would call it DeepSeek; shown as "Inkling 2 Small · UD-Q2_K_XL"; defaults temperature 1.0,
top_p 1.0, no top-k / min-p, effort high). maya.py starts such a config like any other (`./maya.sh --config
maya-inkling2-small.json`, or `run_server`); it has no setup flow for the model yet, so the config is written by hand:

```json
{
 "exe": "/home/shruggr/project-maya/build-hip/maya-inkling2",
 "args": ["--pack", "/mnt/data/models/llm/Inkling-2-Small-GGUF/UD-Q2_K_XL/maya-inkling2", "--max-context", "32768"],
 "cwd": "/home/shruggr/project-maya",
 "tokenizer": "/mnt/data/models/llm/Inkling-2-Small-GGUF/UD-Q2_K_XL/maya-inkling2/tokenizer",
 "model_name": "inkling-2-small",
 "gpu": [0, 1, 2, 3],
 "backend": "hip",
 "lib_dirs": ["/home/shruggr/rocm10-sdk/lib/python3.12/site-packages/_rocm_sdk_devel/lib"],
 "sampling": {"temperature": 1.0, "top_p": 1.0},
 "reasoning_effort": "high",
 "port": 8080,
 "log": "/home/shruggr/project-maya/maya-inkling2-small-hip.log",
 "installer": {"family": "inkling2", "inkling2_quant": "ud-q2_k_xl"}
}
```

Saved as `/home/shruggr/project-maya/maya-<anything>.json` it shows on the Models page; note that maya.py starts the
most recently used config when none is named, so keep it elsewhere (or pass `--config`) until the engine serves.
Engine flags beyond these go in `args` as they are (`--vram-experts`, `--profile`, `--threads`, `--reserve-gb`, ...);
the Models page changes only `--max-context` and the GPUs. `"MAYA_*"` variables go in `"env"`.

The server starts: `Popen([exe, "--serve", *args])` with `args` = the config's args, plus `--layer-split auto` when
`gpu` lists more than one GPU (serve/server.py `engine_args`), in `cwd`, with `HIP_VISIBLE_DEVICES=0,1,2,3` (the
config's GPUs; the engine numbers them 0..n-1) and `lib_dirs` first on the library path. A context change from the
dashboard restarts it with a new `--max-context`. So `maya-inkling2` must take:

- `--serve`; `--max-context N` (today's engine calls it `--ctx`: take both); `--layer-split auto` (or a list);
  `--pack DIR`.

The line protocol (as maya-hy4's, src/hy4/hy4_main.cpp `serve`):

- start: any `NOTE <text>` / `INFO key=value ...` lines (`INFO engine=inkling2 engine_kind=inkling2 model=Inkling-2-Small
  ctx=<n> vram_gb=<x> dspark=1 mtp=0 ...` (`dspark=1`: the drafter is loaded and drafting, the default; `--dspark off`
  or no drafter index: `dspark=0`); `logit_bias=1` / `logprobs=<n>` only if it does those - else the server
  refuses such requests), `VRAM ...` lines optional, then `READY <max_context> stop` (`pause`, `multi=<n>` only if
  supported). stderr is the log.
- the server sends `GEN <max_new> [key=value ...] <id,id,...>` - the whole prompt each time (the engine reuses the
  prefix it already holds); keys: `temperature` (absent = greedy, `temperature=0` greedy), `top_p` (only < 1),
  `top_k` (1..64), `min_p`, `seed`, `penalty_repeat` / `penalty_freq` / `penalty_present` / `penalty_last_n`;
  for the drafter `spec=0|1` (this request drafts or not; the default: it does) and `dspark_cum=F` / `spec_cum=F` (the
  floor on the running product of the drafts' probabilities, default 0.5); unknown
  keys must be ignored (`think_budget` / `think_end` never come for Inkling). `STOP` (end the request now), `WRAP`
  (the server is stopping), `QUIT` (exit). No `GENI` (text only).
- the engine answers `PP <read> <total> <ms> <tok_s> <reused>` while it reads the prompt (also a heartbeat),
  `T <id>` a generated token (flushed each; a round with drafts gives several at once), `STAT tok_s=.. ms_tok=.. ...`
  now and then (with the drafter also `draft_rounds= draft_ms= verify_ms=` and, once drafts were verified,
  `drafts=<kept / offered> draft_acc= draft_off=`, maya-ds4's names), and ends every request with
  `DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <finish> <drafts_acc> <drafts_off> <reused> 0 0`
  (finish `stop` | `length` | `cancel`) - or `ERR <message>` (then a DONE may follow). It stops at its EOS (200006) and
  should stop at 199999 too; ids past 200057 must never be sampled (the padded rows).
- the server reads the T ids with this tokenizer (aliases above), so the engine needs no text at all.

## Open issues

- No thinking budget for Inkling (no end-of-thinking token; would need a 3-token close or an engine-side rule).
- Numeric `reasoning_effort` (0.0-0.99) is not accepted by the server's effort parser (its words only).
- `<|content_invoke_tool_text|>` is read as JSON when it parses; its real format is undocumented here.
- llama.cpp's split differs from the original GPT-4o regex on cased non-ASCII letters (`ÇA`); the server follows
  llama.cpp. Which one Thinking Machines trained with is not checkable without their tokenizer.json.
- The engine as of this writing has no `--serve` and names the context `--ctx`; the server passes `--max-context`.
- No setup flow (download, pack, config) in maya.py, and the start does not compile `maya-inkling2` again after a pull
  (refresh_engine leaves it alone: it is built by hand).
- Vision (`mmproj-F16.gguf`) and the MTP drafter: not served.
