"""tools/test_inkling2_tokenizer.py - Inkling 2 Small's tokenizer (tokenizer.ggml.pre "gpt-4o", llama.cpp's "inkling2"):
the server's tokenizer against llama.cpp's own, id for id.

    python -m unittest tools.test_inkling2_tokenizer          (from the repo root, the .venv's Python)

What is checked:
  * the collapse llama.cpp runs GPT-4o's regex over (every non-ASCII code point one byte for its category; Unicode
    15.1's tables) and the split it gives - "ÇA" is " Ç" + "A" there, one piece in the original regex;
  * the Inkling vocabulary's structural tokens read back as their marks (aliases), the pack's tokenizer files;
  * with the model's tokenizer on this PC (a pack's tokenizer/ folder, or $INKLING2_TOKENIZER): 198 strings tokenized
    by llama.cpp's llama-tokenize (serve/fixtures/inkling2/tokenizer_golden.json: unicode, code, numbers, special tokens
    parsed and not, chat-formatted prompts, fuzz), the port's two reference prompts (tools/inkling2/ref/prompt-*.txt
    against /mnt/data/inkling2-ref/*/tokens.txt), and the round trip;
  * $INKLING2_LLAMA_TOKENIZE (a command that takes a file and prints llama-tokenize's --ids list, e.g. a podman run
    of the reference checkout's llama-tokenize): fresh random strings against it.
Without the model's files the parity tests skip; the rest run anywhere.
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import shlex
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import strata_tokenizer as ST  # noqa: E402

MODEL = pathlib.Path("/mnt/data/models/llm/Inkling-2-Small-GGUF/UD-Q2_K_XL")
GGUF = MODEL / "Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf"
GOLDEN = ROOT / "serve" / "fixtures" / "inkling2" / "tokenizer_golden.json"
REF = pathlib.Path("/mnt/data/inkling2-ref")


def tokenizer_dir() -> pathlib.Path | None:
    for d in ([os.environ["INKLING2_TOKENIZER"]] if os.environ.get("INKLING2_TOKENIZER") else []) + \
            [MODEL / "maya-inkling2" / "tokenizer", MODEL / "maya-inkling2-tokenizer"]:
        p = pathlib.Path(d)
        if (p / "vocab.json").exists() and (p / "merges.txt").exists():
            return p
    return None


_TK = {}


def model_tokenizer() -> ST.Tokenizer:
    d = tokenizer_dir()
    if d is None:
        raise unittest.SkipTest("Inkling 2's tokenizer is not on this PC (a maya-inkling2 pack, or $INKLING2_TOKENIZER)")
    if d not in _TK:
        _TK[d] = ST.Tokenizer.from_dir(d)
    return _TK[d]


def byte_vocab(extra_specials=()) -> tuple[list[str], list[str], list[int]]:
    """A small byte-level vocabulary: the 256 byte characters, a few merges, and the given control tokens."""
    tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
    merges = []
    for a, b in (("Ġ", "t"), ("h", "e"), ("Ġt", "he"), ("C", "O"), ("L", "E"), ("CO", "LE")):
        merges.append(a + " " + b)
        tokens.append(a + b)
    types = [1] * len(tokens)
    for s in extra_specials:
        tokens.append(s)
        types.append(3)
    return tokens, merges, types


INKLING_SPECIALS = ("<|endoftext|>", "<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>",
                    "<|content_text|>", "<|content_model_end_sampling|>", "<|content_thinking|>", "<|end_message|>",
                    "<|content_tool_error|>", "<|content_invoke_tool_json|>", "<|content_invoke_tool_text|>")


class Collapse(unittest.TestCase):
    """llama.cpp's unicode_regex_split for GPT-4o's regex: the text collapsed by category, the pattern over that."""

    def test_categories(self):
        t = ST._Collapse()
        for ch, want in (("A", "A"), ("z", "z"), ("7", "7"), (" ", " "), ("\x1c", "\x1c"), ("É", "\xd2"), ("ß", "\xd2"),
                         ("日", "\xd2"), ("½", "\xd1"), ("٣", "\xd1"), ("́", "\xd4"), ("€", "\xd5"), ("—", "\xd3"),
                         (" ", "\x0b"), ("\u0085", "\x0b"), ("　", "\x0b"), (" ", "\x0b"),
                         ("​", "\xd0"), ("", "\xd0"), ("\U00010d40", "\xd0"), ("ࢗ", "\xd0")):
            self.assertEqual(t[ord(ch)], want, repr(ch))

    def test_unicode_16_overlay_parses(self):
        rs = ST._ranges(ST._LLAMA_UNDEFINED)
        self.assertEqual(len(rs), 47)
        self.assertTrue(all(a <= b for a, b in rs))
        self.assertEqual(rs[0], (0x897, 0x897))

    def test_split_differs_from_the_original_regex_on_cased_non_ascii(self):
        llama = ST._LlamaCollapsed(ST.GPT4O_COLLAPSED_PATTERN)
        import regex
        orig = regex.compile(ST.MINIMAX_PATTERN)
        for text, want_llama, want_orig in (
                ("ÇA VA", ["Ç", "A", " VA"], ["ÇA", " VA"]),
                ("ÉCOLE", ["É", "COLE"], ["ÉCOLE"]),
                ("Hello World", ["Hello", " World"], ["Hello", " World"]),
                ("HTTPServer", ["HTTPServer"], ["HTTPServer"]),
                ("I'M you're", ["I'M", " you're"], ["I'M", " you're"]),
                ("1234567", ["123", "456", "7"], ["123", "456", "7"]),
                ("a  b\n\n c", ["a", " ", " b", "\n\n", " c"], ["a", " ", " b", "\n\n", " c"]),
                ("x\x1cy", ["x", "\x1cy"], ["x", "\x1cy"])):
            self.assertEqual(llama.findall(text), want_llama, text)
            self.assertEqual(orig.findall(text), want_orig, text)

    def test_every_character_is_in_a_piece(self):
        llama = ST._LlamaCollapsed(ST.GPT4O_COLLAPSED_PATTERN)
        rng = random.Random(5)
        for _ in range(300):
            s = "".join(chr(rng.choice([rng.randrange(0x20, 0x7f), rng.randrange(0xa0, 0x3000), rng.randrange(0, 0x20),
                                        rng.randrange(0x1f300, 0x1fa00)])) for _ in range(rng.randrange(1, 40)))
            self.assertEqual("".join(llama.findall(s)), s)


class Vocabulary(unittest.TestCase):
    """The pre-tokenizer's names, the aliases of Inkling's structural tokens, the pack's tokenizer files."""

    def test_names_and_regex_choice(self):
        tokens, merges, types = byte_vocab()
        for pre in ("gpt-4o", "inkling2"):
            tk = ST.Tokenizer(tokens, merges, types, pre)
            self.assertEqual(tk.pre_regex, "llama")
            self.assertIsInstance(tk._re, ST._LlamaCollapsed)
            self.assertEqual(tk.encode("COLE the"), [tk.ids["COLE"], tk.ids["Ġthe"]])
        tk = ST.Tokenizer(tokens, merges, types, "inkling2", pre_regex="unicode")
        self.assertNotIsInstance(tk._re, ST._LlamaCollapsed)
        with self.assertRaises(ValueError):
            ST.Tokenizer(tokens, merges, types, "inkling2", pre_regex="tiktoken")
        self.assertIsNone(ST.Tokenizer(tokens, merges, types, "qwen35").pre_regex)   # (other models: as before)

    def test_aliases_only_for_inklings_vocabulary(self):
        tokens, merges, types = byte_vocab(INKLING_SPECIALS)
        tk = ST.Tokenizer(tokens, merges, types, "gpt-4o")
        ids = {t: i for i, t in enumerate(tokens)}
        self.assertEqual(tk.aliases, {ids[t]: m for t, m in ST.INKLING2_MARKS.items()})
        self.assertEqual(tk.token_bytes(ids["<|message_model|>"]), "\U0010fff0".encode())
        self.assertEqual(tk.token_bytes(ids["<|content_model_end_sampling|>"]), b"<|content_model_end_sampling|>")
        # prompts still encode the tokens by their own text
        self.assertEqual(tk.encode("<|message_model|><|content_text|>", parse_special=True),
                         [ids["<|message_model|>"], ids["<|content_text|>"]])
        # another gpt-4o vocabulary (Llama 4's, ...): none
        tokens2, merges2, types2 = byte_vocab(("<|endoftext|>", "<|message_model|>"))
        self.assertEqual(ST.Tokenizer(tokens2, merges2, types2, "gpt-4o").aliases, {})
        self.assertEqual(ST.Tokenizer(tokens, merges, types, "qwen35").aliases, {})

    def test_marks_match_the_frontends(self):
        from serve import frontend
        self.assertEqual(ST.INKLING2_MARKS, frontend.INKLING_MARKS)

    def test_tokenizer_files_and_from_dir(self):
        tokens, merges, types = byte_vocab(INKLING_SPECIALS)
        md = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "gpt-4o", "tokenizer.ggml.tokens": tokens,
              "tokenizer.ggml.merges": merges, "tokenizer.ggml.token_type": types,
              "tokenizer.ggml.bos_token_id": tokens.index("<|content_model_end_sampling|>"),
              "tokenizer.ggml.eos_token_id": tokens.index("<|content_model_end_sampling|>"),
              "tokenizer.ggml.add_bos_token": False, "tokenizer.chat_template": "{{ messages }}\n"}
        files = ST.tokenizer_files(md)
        self.assertEqual(set(files), {"vocab.json", "merges.txt", "token_type.json", "tokenizer.json",
                                      "chat_template.jinja"})
        cfg = json.loads(files["tokenizer.json"])
        eos = tokens.index("<|content_model_end_sampling|>")
        self.assertEqual((cfg["pre"], cfg["pre_regex"], cfg["special_ids"], cfg["add_bos_token"]),
                         ("gpt-4o", "llama", {"bos": eos, "eos": eos}, False))
        self.assertEqual(cfg["pre_pattern"], ST.GPT4O_LLAMA_PATTERN)
        self.assertEqual(len(cfg["aliases"]), len(ST.INKLING2_MARKS))
        with tempfile.TemporaryDirectory() as d:
            for name, text in files.items():
                pathlib.Path(d, name).write_text(text, encoding="utf-8")
            tk = ST.Tokenizer.from_dir(d)
            self.assertEqual((tk.pre, tk.special_ids["eos"]), ("gpt-4o", eos))
            self.assertEqual(tk.aliases, ST.inkling2_aliases(tokens, types))
            # a pack written without the aliases (or the regex choice) gets them all the same
            cfg.pop("aliases")
            cfg["pre_regex"] = "unicode"
            pathlib.Path(d, "tokenizer.json").write_text(json.dumps(cfg), encoding="utf-8")
            tk = ST.Tokenizer.from_dir(d)
            self.assertEqual((tk.aliases, tk.pre_regex), (ST.inkling2_aliases(tokens, types), "unicode"))


class Parity(unittest.TestCase):
    """The model's own tokenizer against llama.cpp's ids."""

    def test_golden_cases(self):
        tk = model_tokenizer()
        cases = json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]
        self.assertGreaterEqual(len(cases), 150)
        bad = []
        for case in cases:
            text, special = case[0], case[1]
            raw = case[2] if len(case) > 2 else special
            for parse, want in ((True, special), (False, raw)):
                got = tk.encode(text, parse_special=parse)
                if got != want:
                    bad.append((text[:40], parse, len(want), len(got)))
        self.assertEqual(bad, [])

    def test_reference_prompts(self):
        tk = model_tokenizer()
        n = 0
        for name, want_len in (("short", 12), ("long", 2698)):
            ref = REF / name / "tokens.txt"
            if not ref.exists():
                continue
            want = [int(x) for x in ref.read_text().splitlines()[0].split()]
            text = (HERE / "inkling2" / "ref" / f"prompt-{name}.txt").read_text(encoding="utf-8")
            self.assertEqual(len(want), want_len)
            self.assertEqual(tk.encode(text, parse_special=True), want, name)
            n += 1
        if not n:
            self.skipTest("no reference dumps here (/mnt/data/inkling2-ref)")

    def test_round_trip_and_specials(self):
        tk = model_tokenizer()
        cases = json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]
        for case in cases:
            self.assertEqual(tk.decode(tk.encode(case[0])), case[0])
        ids = {s: tk.special_tokens[s] for s in ("<|endoftext|>", "<|message_user|>", "<|message_model|>",
                                                  "<|message_system|>", "<|message_tool|>", "<|content_text|>",
                                                  "<|content_model_end_sampling|>", "<|content_thinking|>",
                                                  "<|end_message|>", "<|content_invoke_tool_json|>")}
        self.assertEqual(ids, {"<|endoftext|>": 199999, "<|message_user|>": 200000, "<|message_model|>": 200001,
                               "<|message_system|>": 200002, "<|message_tool|>": 200003, "<|content_text|>": 200004,
                               "<|content_model_end_sampling|>": 200006, "<|content_thinking|>": 200008,
                               "<|end_message|>": 200010, "<|content_invoke_tool_json|>": 200049})
        self.assertEqual(len(tk.tokens), 200058)
        self.assertEqual(tk.special_ids.get("eos"), 200006)
        self.assertEqual(tk.token_bytes(200010), "\U0010fff5".encode())          # (the server's reading)
        self.assertEqual(tk.decode([200006]), "<|content_model_end_sampling|>")

    def test_pack_folder_matches_the_gguf(self):
        tk = model_tokenizer()
        if not GGUF.exists():
            self.skipTest("the GGUF is not on this PC")
        from gguf_reader import GGUFFile
        files = ST.tokenizer_files(GGUFFile(GGUF).metadata)
        vocab = json.loads(files["vocab.json"])
        self.assertEqual(len(vocab), len(tk.tokens))
        self.assertTrue(all(tk.tokens[i] == s for s, i in vocab.items()))
        self.assertEqual(len(files["merges.txt"].split("\n")), len(tk.ranks))
        self.assertEqual(json.loads(files["tokenizer.json"])["special_ids"], {"eos": 200006, "bos": 200006})


@unittest.skipUnless(os.environ.get("INKLING2_LLAMA_TOKENIZE"), "set INKLING2_LLAMA_TOKENIZE to llama-tokenize's command")
class LiveOracle(unittest.TestCase):
    """Fresh random strings against llama.cpp itself: $INKLING2_LLAMA_TOKENIZE is a command line that gets a file's path
    appended and prints llama-tokenize's `--ids` list (with --no-escape; special tokens parsed)."""

    def test_random_strings(self):
        tk = model_tokenizer()
        cmd = shlex.split(os.environ["INKLING2_LLAMA_TOKENIZE"])
        rng = random.Random(int(os.environ.get("INKLING2_SEED", "1")))
        words = ["ÉCOLE", "École", "ÇA", "naïve", "МОСКВА", "Москва", "I'M", "don't", "123456", "x²", "東京", "😀",
                 "á", "HTTPServer", " ", "\n", "\t", "  ", "<|end_message|>", "<|message_", "١٢٣", "ǅ"]
        with tempfile.TemporaryDirectory() as d:
            for k in range(int(os.environ.get("INKLING2_CASES", "20"))):
                s = "".join(rng.choice(words) + rng.choice(["", " ", ".", "\n"]) for _ in range(rng.randrange(1, 20)))
                p = pathlib.Path(d, f"c{k}.txt")
                p.write_bytes(s.encode("utf-8"))
                out = subprocess.run(cmd + [str(p)], capture_output=True, text=True, check=True).stdout
                want = json.loads(out.strip().splitlines()[-1])
                self.assertEqual(tk.encode(s, parse_special=True), want, repr(s))


if __name__ == "__main__":
    unittest.main()
