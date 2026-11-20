# Inkling 2 Small reference files

- `chat_template.jinja` - the chat template inside unsloth's `Inkling-2-Small-UD-Q2_K_XL` GGUF set
  (`tokenizer.chat_template` of shard 1, sha256 `d4c852bf15f2c08a1592d2024755f86427de75bc5e5e3272cd98da74ecb3475f`):
  Thinking Machines' Inkling 2 template as llama.cpp's conversion shipped it (typed message blocks, canonical JSON,
  "Thinking effort level" block, no generation prefix). It is what a `maya-inkling2` pack's `tokenizer/` folder holds
  and what the server renders. The template is Thinking Machines'; see the model's page for its terms.
- `tokenizer_golden.json` - 198 strings and the ids llama.cpp's `llama-tokenize` (checkout b94959a64) gives them, with
  special tokens parsed and not (`tools/test_inkling2_tokenizer.py`).
- `llama_chat_golden.json` - 18 conversations rendered and 10 model answers parsed by llama.cpp's own chat code
  (`common_chat_templates_apply`, `common_chat_session`) - `serve/test_inkling2_frontend.py` compares the server's.
  Made with `ink_chat.cpp` (built in the ROCm container against the reference checkout's build-hip libraries:
  `ink_chat_build.sh`, with the folder mounted at /work) on the cases `ink_chat_cases.py` writes.
