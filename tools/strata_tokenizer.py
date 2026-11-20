"""tools/strata_tokenizer.py - the tokenizer, as the reference implementation of record.

WHY THIS EXISTS AND WHERE IT SITS.  The GGUF carries the whole tokenizer in metadata: `tokenizer.ggml.model`
= `gpt2`, 248,320 `tokens`, 247,587 `merges`, `token_type`, `pre` = `qwen35`, and a Jinja `chat_template`.
None of that needs the 35 GB of weights, so it is extracted into the pack's `tokenizer/` directory
(docs/pack-format.md §6) and this module is what reads it back.  A C++ port follows and is tested against
this one.

THE DECISIVE PROPERTY IS THE ROUND TRIP.  Byte-level BPE maps each BYTE to a printable unicode character so
that any UTF-8 input is representable with no UNK token.  Every failure mode of that mapping - the wrong
offset for a byte, a merge applied in the wrong order, a pre-tokenizer split that drops a character - still
produces plausible token ids.  `decode(encode(s)) == s` is the check that sees them, and it is exact because
the byte layer is lossless.  It is asserted over a corpus chosen to hit the boundaries: multi-byte UTF-8,
emoji (4-byte), combining marks, whitespace runs, and C0 control bytes.
"""
from __future__ import annotations

import json
import pathlib
import sys

import regex

# ------------------------------------------------------------------ byte <-> unicode (GPT-2 byte-level BPE)
def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte encoder: 256 bytes -> 256 printable characters, reversibly.

    Bytes 33..126, 161..172 and 174..255 map to themselves; the remaining 68 (space, newline, the C0
    controls and the high range that would be invisible) are shifted into 256+n so that no byte is
    unprintable.  The shift is the whole trick and getting its ORDER wrong is invisible in the output ids.
    """
    bs = (list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}

# The `qwen35` pre-tokenizer, transcribed from the ORACLE rather than from the family resemblance:
# `.ref/llama.cpp/src/llama-vocab.cpp` L396, `case LLAMA_VOCAB_PRE_TYPE_QWEN35`.  The commented-out line
# above it is the `tokenizer.json` original, and llama.cpp's active version differs from it - it spells the
# contraction classes out instead of using `(?i:...)`, which is behaviourally the same.
#
# THE `\p{M}` IS THE WHOLE POINT OF THIS BEING TRANSCRIBED.  The QWEN3 pattern two cases earlier (L389) is
# the same shape with `\p{L}` where this has `[\p{L}\p{M}]` and without `\p{M}` in the punctuation negation.
# Writing the QWEN3 pattern for `qwen35` looks right, round-trips perfectly, and is wrong: a combining mark
# is `\p{M}`, so without it the mark gets swallowed into the following punctuation run and a token like `_j`
# never forms.  It cost 3 strings out of 1875 and only `tokenize_oracle_check` could see it.
QWEN35_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# The `glm4`/`glm5` pre-tokenizer, transcribed from llama-vocab.cpp L406-410 at the vendored pin
# (LLAMA_VOCAB_PRE_TYPE_CHATGLM4).  Two differences from qwen35, both load-bearing:
#   * the digit run is `\p{N}{1,3}` - a 4-digit number splits 3+1, not as one whole run;
#   * the letter class is `\p{L}+` WITHOUT `\p{M}`, AND the punctuation class is
#     `[^\s\p{L}\p{N}]` without `\p{M}` - so a combining mark attaches to the FOLLOWING
#     punctuation run, the exact opposite of the qwen35 `\p{M}` subtlety above.
# AND the encoding semantic differs: llama.cpp sets `ignore_merges = true` for glm4/glm5, so a
# pre-token piece that is an exact vocabulary token is emitted WHOLE (no merges applied); only
# the leftovers go through BPE.  Transcribing the regex alone and keeping the merges would be
# the same class of plausible-but-wrong as the qwen3-for-qwen35 mistake above.
GLM_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


# The `deepseek3` pre-tokenizer (DeepSeek-V3 .. V4.1's tokenizer.json, llama.cpp's LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM):
# THREE Split stages with behavior "Isolated", each applied to every piece the one before produced - digit runs of at
# most 3, then runs of CJK ideographs / kana, then the word pattern - and the text between a stage's matches stays a
# piece of its own (a digit run passes the word pattern untouched that way).  Then the byte-level mapping; merges by
# rank as usual (no ignore_merges).
DEEPSEEK3_PATTERNS = (
    r"\p{N}{1,3}",
    r"[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",
    r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+"
    r"| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
)
# `hy_v4` (Hy4-flash's tokenizer) splits with these same three stages, word for word (the HUNYUAN_DENSE expressions,
# which are DEEPSEEK3_LLM's), merges by rank
PRE_ISOLATED = {"deepseek3": DEEPSEEK3_PATTERNS, "hy_v4": DEEPSEEK3_PATTERNS}


# The `minimax` pre-tokenizer: MiniMax-M3.1's tokenizer.json verbatim - an NFC normalizer, then one Split (invert,
# "Removed") with this regex, GPT-4o's (o200k) - before the byte-level map.  The NFC runs on each plain segment,
# between the added tokens, as `tokenizers` does for added tokens that are not normalized.
MINIMAX_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# The `gpt-4o` pre-tokenizer (Inkling 2's GGUF: tokenizer.ggml.pre "gpt-4o"; llama.cpp's converter now writes
# "inkling2" for the same tokenizer - both are LLAMA_VOCAB_PRE_TYPE_GPT4O, llama-vocab.cpp ~L455) AS LLAMA.CPP RUNS IT.
# The regex is GPT-4o's (o200k: MINIMAX_PATTERN above, the tokenizer.json original), but llama.cpp has no custom
# splitter for it and takes its std::regex path: every non-ASCII code point is first collapsed to one byte for its
# category (letter 0xD2, number 0xD1, punctuation 0xD3, mark 0xD4, symbol 0xD5, whitespace 0x0B, anything else 0xD0),
# \p{L} becomes [\xD2A-Za-z] (and so on), and - because a collapsed letter has no case - the case classes are spelled
# with lookaheads: [\p{Lu}..\p{M}]* is "a letter or mark that is not a-z", [\p{Ll}..\p{M}]+ "... not A-Z".  So for a
# cased non-ASCII letter llama.cpp's split differs from the original regex: the original keeps "ÇA" whole ("Ç" is upper
# case), llama.cpp splits "Ç" + "A" (the collapsed "Ç" is both cases, and the first alternative wins with it alone) -
# and BPE then gives other ids for " ÇA" (in 1,876 test strings the only difference; "ÉCOLE" splits too, same ids).  The
# server's tokenizer follows llama.cpp (the port's reference, tools/test_inkling2_tokenizer.py checks it id for id);
# a pack's tokenizer.json "pre_regex": "unicode" switches to the original regex (tokenizers' / tiktoken's semantics).
# `\s` is std::regex's (ASCII space, \t \n \v \f \r; not \x1c-\x1f, which Python's \s also matches).
_LL, _LN, _LM, _LS = "\xd2A-Za-z", "\xd10-9", "\xd4", " \\t\\n\\x0b\\x0c\\r"
_CONTRACTIONS = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?"
GPT4O_COLLAPSED_PATTERN = (
    "[^\\r\\n" + _LL + _LN + "]?(?:(?=[" + _LL + _LM + "])[^a-z])*(?:(?=[" + _LL + _LM + "])[^A-Z])+" + _CONTRACTIONS +
    "|[^\\r\\n" + _LL + _LN + "]?(?:(?=[" + _LL + _LM + "])[^a-z])+(?:(?=[" + _LL + _LM + "])[^A-Z])*" + _CONTRACTIONS +
    "|[" + _LN + "]{1,3}"
    "| ?[^" + _LS + _LL + _LN + "]+[\\r\\n/]*"
    "|[" + _LS + "]*[\\r\\n]+"
    "|[" + _LS + "]+(?![^" + _LS + "])"
    "|[" + _LS + "]+"
)
# (the pattern llama.cpp compiles before the collapse, as shipped in a pack's tokenizer.json for a C++ port)
GPT4O_LLAMA_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]"
    r"|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]"
    r"|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
GPT4O_PRES = ("gpt-4o", "inkling2")
# llama.cpp's own whitespace set (scripts/gen-unicode-data.py: PropList's White_Space) - collapsed to 0x0B
_LLAMA_SPACE = frozenset([0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000, *range(0x2000, 0x200B)])
_LLAMA_CATEGORY = {"L": "\xd2", "N": "\xd1", "P": "\xd3", "M": "\xd4", "S": "\xd5"}
# llama.cpp's unicode-data.cpp (b94959a64) is Unicode 15.1; Python's unicodedata here is 16.0.  The code points 16.0
# added are "undefined" (0xD0) there - kept so, or a letter of Garay or Todhri would split differently.
_LLAMA_UNDEFINED = (
    "897,1B4E-1B4F,1B7F,1C89-1C8A,2427-2429,31E4-31E5,A7CB-A7CD,A7DA-A7DC,105C0-105F3,10D40-10D65,10D69-10D85,"
    "10D8E-10D8F,10EC2-10EC4,10EFC,11380-11389,1138B,1138E,11390-113B5,113B7-113C0,113C2,113C5,113C7-113CA,"
    "113CC-113D5,113D7-113D8,113E1-113E2,116D0-116E3,11BC0-11BE1,11BF0-11BF9,11F5A,13460-143FA,16100-16139,"
    "16D40-16D79,18CFF,1CC00-1CCF9,1CD00-1CEB3,1E5D0-1E5FA,1E5FF,1F8B2-1F8BB,1F8C0-1F8C1,1FA89,1FA8F,1FABE,1FAC6,"
    "1FADC,1FADF,1FAE9,1FBCB-1FBEF")


def _ranges(spec: str) -> list[tuple[int, int]]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.append((int(a, 16), int(b or a, 16)))
    return out


class _Collapse(dict):
    """str.translate table: a code point -> its llama.cpp collapsed character (ASCII stays itself), filled lazily."""

    def __init__(self):
        super().__init__()
        # (another Python's tables: the 16.0 additions stay undefined; later ones take Python's category)
        self.undefined = _ranges(_LLAMA_UNDEFINED)

    def __missing__(self, cp: int) -> str:
        import unicodedata
        if cp < 128:
            ch = chr(cp)
        elif cp in _LLAMA_SPACE:
            ch = "\x0b"
        elif any(a <= cp <= b for a, b in self.undefined):
            ch = "\xd0"
        else:
            ch = _LLAMA_CATEGORY.get(unicodedata.category(chr(cp))[0], "\xd0")
        if len(self) < 200_000:                          # (remembered: at most this many code points)
            self[cp] = ch
        return ch


class _LlamaCollapsed:
    """findall() as llama.cpp's unicode_regex_split does it for a pattern with unicode categories and no custom
    splitter: the pattern over the collapsed text (one character per code point, so the offsets are the text's own),
    the text between two matches a piece of its own (unicode_regex_split_stl)."""

    _table = None

    def __init__(self, pattern: str):
        self.pat = regex.compile(pattern)
        if _LlamaCollapsed._table is None:
            _LlamaCollapsed._table = _Collapse()

    def findall(self, text: str) -> list[str]:
        collapsed = text.translate(self._table)
        out, pos = [], 0
        for m in self.pat.finditer(collapsed):
            if m.start() > pos:
                out.append(text[pos:m.start()])
            out.append(text[m.start():m.end()])
            pos = m.end()
        if pos < len(text):
            out.append(text[pos:])
        return out


# Inkling 2's answer is a list of blocks, each opened by <|message_model|> and a content token and closed by
# <|end_message|>.  The server reads the tokens that structure it (and only those: their TEXT written by the model out
# of ordinary pieces stays text) as one private-use character each - the tokenizer's "aliases" - which
# serve/frontend.py's InklingRewriter takes apart (KEEP IN STEP: frontend.INKLING_MARKS).
INKLING2_MARKS = {"<|message_model|>": "\U0010fff0", "<|content_thinking|>": "\U0010fff1",
                  "<|content_text|>": "\U0010fff2", "<|content_invoke_tool_json|>": "\U0010fff3",
                  "<|content_invoke_tool_text|>": "\U0010fff4", "<|end_message|>": "\U0010fff5",
                  "<|content_tool_error|>": "\U0010fff6"}


def inkling2_aliases(tokens: list[str], token_types: list[int] | None) -> dict[int, str]:
    """{id: mark} for Inkling 2's structural tokens when this vocabulary is Inkling's (its <|message_model|> and
    <|content_model_end_sampling|> control tokens), else {}."""
    if not token_types:
        return {}
    ids = {t: i for i, t in enumerate(tokens) if t.startswith("<|") and token_types[i] == 3}
    if "<|message_model|>" not in ids or "<|content_model_end_sampling|>" not in ids:
        return {}
    return {ids[t]: mark for t, mark in INKLING2_MARKS.items() if t in ids}


class _IsolatedSplits:
    """findall() over a sequence of "Isolated" splits (see DEEPSEEK3_PATTERNS)."""

    def __init__(self, patterns):
        self.stages = [regex.compile(p) for p in patterns]

    def findall(self, text: str) -> list[str]:
        pieces = [text]
        for pat in self.stages:
            out = []
            for piece in pieces:
                pos = 0
                for m in pat.finditer(piece):
                    if m.start() > pos:
                        out.append(piece[pos:m.start()])
                    if m.end() > m.start():
                        out.append(m.group(0))
                    pos = m.end()
                if pos < len(piece):
                    out.append(piece[pos:])
            pieces = out
        return pieces


class Tokenizer:
    def __init__(self, tokens: list[str], merges: list[str], token_types: list[int] | None = None,
                 pre: str = "qwen35", special_ids: dict[str, int] | None = None, pre_regex: str = "llama"):
        self.tokens = tokens
        self.pre = pre
        # Each supported scheme names a transcription of the vendored llama.cpp reference; an
        # unknown scheme is REFUSED rather than silently tokenized with the wrong splits.
        if pre not in ("qwen35", "glm4", "glm5", "deepseek3", "minimax", "hy_v4") + GPT4O_PRES:
            raise ValueError("the %r pre-tokenizer is not transcribed; known: qwen35, glm4, glm5, deepseek3, minimax, "
                             "hy_v4, gpt-4o, inkling2" % pre)
        if pre_regex not in ("llama", "unicode"):
            raise ValueError("pre_regex %r: llama or unicode" % pre_regex)
        # gpt-4o / inkling2: llama.cpp's collapsed split (GPT4O_COLLAPSED_PATTERN), or the original regex ("unicode")
        self.pre_regex = pre_regex if pre in GPT4O_PRES else None
        # glm4/glm5 set ignore_merges: an exact vocab hit emits the whole pre-token piece
        self.ignore_merges = pre in ("glm4", "glm5")
        self.token_types = token_types
        self.special_ids = special_ids or {}
        self.ids = {t: i for i, t in enumerate(tokens)}
        if len(self.ids) != len(tokens):
            raise ValueError("vocabulary has duplicate tokens: %d entries, %d unique"
                             % (len(tokens), len(self.ids)))
        # Merge rules as (left, right) -> rank.  A merge list that is not a total order over its pairs, or one
        # naming a token that is not in the vocabulary, would make BPE silently produce different ids than the
        # model was trained with, so both are checked here rather than discovered as a bad answer later.
        self.ranks: dict[tuple[str, str], int] = {}
        for i, m in enumerate(merges):
            parts = m.split(" ")
            if len(parts) != 2:
                raise ValueError("merge %d is not a pair: %r" % (i, m))
            if parts[0] not in self.ids or parts[1] not in self.ids:
                raise ValueError("merge %d names a token outside the vocabulary: %r" % (i, m))
            self.ranks[(parts[0], parts[1])] = i
        self._re = (_IsolatedSplits(PRE_ISOLATED[pre]) if pre in PRE_ISOLATED else
                    regex.compile(MINIMAX_PATTERN) if pre == "minimax" or self.pre_regex == "unicode" else
                    _LlamaCollapsed(GPT4O_COLLAPSED_PATTERN) if pre in GPT4O_PRES else
                    regex.compile(GLM_PATTERN if self.ignore_merges else QWEN35_PATTERN))
        self._nfc = pre == "minimax"
        # the server's reading of some output tokens (token_bytes; see from_dir): Inkling 2's structural tokens
        self.aliases = inkling2_aliases(tokens, token_types) if pre in GPT4O_PRES else {}

        # The literals matched directly instead of being run through BPE.  GGUF token types: 3 = CONTROL,
        # 4 = USER_DEFINED.  The two classes behave DIFFERENTLY and llama.cpp's own tokenizer settled which:
        #
        #   * type 4 (USER_DEFINED: `<think>`, `<tool_call>`, `<tool_response>`, ...) is matched ALWAYS,
        #     with or without parse_special.
        #   * type 3 (CONTROL: `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, ...) is matched ONLY when
        #     parse_special is set.
        #
        # Measured, not assumed: with parse_special=False the oracle still emitted `<tool_response>` as one
        # token and mine emitted four, and the three disagreements were exactly the type-4 cases while every
        # type-3 case agreed.  Treating both classes alike costs 3 strings in 1229 and silently changes a chat
        # prompt, because `<|im_end|>` decomposed into ordinary pieces is not the token the model expects.
        self.special_tokens: dict[str, int] = {}
        if token_types:
            for i, ty in enumerate(token_types):
                if ty in (3, 4):
                    self.special_tokens[tokens[i]] = i
        always = [t for t, i in self.special_tokens.items() if token_types and token_types[i] == 4]
        # Longest literal first, or `<|im_end|>` could match a shorter prefix of itself.  `regex.escape` so a
        # token containing regex metacharacters (several do: `<|`, `[`, `(`) is matched literally.
        self._always_re = self._alt(always)
        self._special_re = self._alt(list(self.special_tokens))

    @staticmethod
    def _alt(literals: list[str]):
        if not literals:
            return None
        return regex.compile("|".join(regex.escape(s) for s in sorted(literals, key=len, reverse=True)))

    # -------------------------------------------------------------- constructors
    @classmethod
    def from_dir(cls, path) -> "Tokenizer":
        """A pack's tokenizer/ directory (extract() below), with the pre-tokenizer its tokenizer.json names: built
        without it, a GLM pack was split the qwen35 way - every number one digit a token where GLM groups up to three
        (#27: numbers came back garbled in long prompts).  No tokenizer.json (never written by extract): qwen35."""
        t = pathlib.Path(path)
        vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for s, i in vocab.items():
            tokens[i] = s
        meta = json.loads((t / "tokenizer.json").read_text(encoding="utf-8")) if (t / "tokenizer.json").exists() else {}
        types = json.loads((t / "token_type.json").read_text()) if (t / "token_type.json").exists() else None
        tk = cls(tokens, (t / "merges.txt").read_text(encoding="utf-8").split("\n"), types,
                 meta.get("pre", "qwen35"), meta.get("special_ids") or None, meta.get("pre_regex") or "llama")
        # "aliases" {id: text}: the text an output token stands for in what the server reads (token_bytes) - Hy4-flash's
        # suffixed tags under GLM's spelling (<think:6124c78e> -> <think>, its <tool_calls:..> wrapper -> nothing),
        # Inkling 2's structural tokens as one private-use character each (INKLING2_MARKS: also without the key);
        # prompts still encode the tokens by their own text
        tk.aliases = {**tk.aliases, **{int(k): v for k, v in (meta.get("aliases") or {}).items()}}
        return tk

    @classmethod
    def from_gguf(cls, path) -> "Tokenizer":
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from gguf_reader import GGUFFile
        md = GGUFFile(pathlib.Path(path)).metadata
        need = ["tokenizer.ggml.tokens", "tokenizer.ggml.merges"]
        missing = [k for k in need if k not in md]
        if missing:
            raise ValueError("GGUF is missing %s; it does not carry a tokenizer" % ", ".join(missing))
        model = md.get("tokenizer.ggml.model")
        if model != "gpt2":
            raise ValueError("expected a byte-level BPE tokenizer (model 'gpt2'), got %r" % model)
        special = {k: int(md[k]) for k in md
                   if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")}
        return cls(list(md["tokenizer.ggml.tokens"]), list(md["tokenizer.ggml.merges"]),
                   list(md.get("tokenizer.ggml.token_type") or []) or None,
                   md.get("tokenizer.ggml.pre", "qwen35"), special)

    # -------------------------------------------------------------- the algorithm
    def _bpe(self, word: str) -> list[str]:
        """Merge `word` (already byte-mapped) by LOWEST RANK first, repeatedly - not left to right.

        Applying merges in list order rather than rank order is the classic BPE bug: it produces a different
        segmentation and a plausible token count.
        """
        parts = list(word)
        while len(parts) > 1:
            best, best_rank = None, None
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            parts[best:best + 2] = [parts[best] + parts[best + 1]]
        return parts

    def _encode_plain(self, text: str) -> list[int]:
        # a piece always encodes to the same ids, so they are remembered (a prompt repeats most of its words, and an
        # agent re-sends the whole conversation every turn); the memo is dropped whole when it grows past ~300K pieces
        memo = self.__dict__.setdefault("_piece_memo", {})
        if len(memo) > 300_000:
            memo.clear()
        out: list[int] = []
        if self.__dict__.get("_nfc"):
            import unicodedata
            text = unicodedata.normalize("NFC", text)
        for piece in self._re.findall(text):
            hit = memo.get(piece)
            if hit is not None:
                out.extend(hit)
                continue
            # The vocab is stored in the BYTE-LEVEL alphabet, so the ignore_merges whole-vocab
            # lookup (llama-vocab.cpp L633-637: an exact hit is emitted whole, only leftovers
            # reach BPE) must run on the byte-MAPPED piece - for ASCII the mapping is the
            # identity and the raw lookup agrees, for everything else it does not (x=Δy²: the
            # mapped piece 'Â²' is one token, the raw piece '²' is not in the vocab at all).
            mapped = "".join(BYTE_TO_UNICODE[b] for b in piece.encode("utf-8"))
            if self.ignore_merges and mapped in self.ids:
                ids = (self.ids[mapped],)
            else:
                got = []
                for tok in self._bpe(mapped):
                    i = self.ids.get(tok)
                    if i is None:
                        raise KeyError("BPE produced a token outside the vocabulary: %r" % tok)
                    got.append(i)
                ids = tuple(got)
            memo[piece] = ids
            out.extend(ids)
        return out

    def _encode_segment(self, text: str) -> list[int]:
        """_encode_plain, remembering long segments (the text between two special tokens: a message, a tool result):
        BPE never merges across a special token, so a re-sent conversation's earlier messages encode to the same ids
        and come from here (LRU, up to ~1M tokens)."""
        if len(text) < 2048:
            return self._encode_plain(text)
        cache = self.__dict__.get("_seg_cache")
        if cache is None:
            import collections
            import threading
            cache = self.__dict__["_seg_cache"] = collections.OrderedDict()
            self.__dict__["_seg_lock"] = threading.Lock()
            self.__dict__["_seg_tokens"] = 0
        lock = self.__dict__["_seg_lock"]
        with lock:
            hit = cache.get(text)
            if hit is not None:
                cache.move_to_end(text)
                return list(hit)
        ids = self._encode_plain(text)
        with lock:
            if text not in cache:
                cache[text] = tuple(ids)
                self.__dict__["_seg_tokens"] += len(ids)
                while self.__dict__["_seg_tokens"] > 1_000_000 and len(cache) > 1:
                    _, old = cache.popitem(last=False)
                    self.__dict__["_seg_tokens"] -= len(old)
        return ids

    def _encode_matching(self, text: str, pat, plain=()) -> list[int]:
        """Encode `text`, emitting any literal `pat` matches as single tokens and BPE-ing the rest.

        KEEP IN STEP: serve/server.py's PromptMemo._encode (MAYA_PROMPT_MEMO) runs a copy of this loop over a prompt's
        tail, using `_special_re`, `special_tokens` and `_encode_segment`; a change here must be made there too
        (serve/test_prompt_memo.py checks that both give the same ids).

        The split happens on the RAW text, before the byte mapping, because a special token's string is a
        literal to match rather than bytes to decompose.  Everything between the matches is tokenized
        normally - which is why a near-miss like `<|im_star` still costs ordinary tokens.  A match that starts
        inside one of the `plain` (start, end) spans is not a token: it is left to the text around it.
        """
        if pat is None:
            return self._encode_segment(text)
        out: list[int] = []
        pos = 0
        spans = iter(sorted(plain))
        upcoming = next(spans, None)
        plain_end = 0
        for m in pat.finditer(text):
            start = m.start()
            while upcoming is not None and upcoming[0] <= start:   # matches come in text order: each span once
                plain_end = max(plain_end, upcoming[1])
                upcoming = next(spans, None)
            if start < plain_end:
                continue
            if start > pos:
                out.extend(self._encode_segment(text[pos:start]))
            out.append(self.special_tokens[m.group(0)])
            pos = m.end()
        if pos < len(text):
            out.extend(self._encode_segment(text[pos:]))
        return out

    def encode(self, text: str, parse_special: bool = False, plain=()) -> list[int]:
        """Tokenize `text`.

        `parse_special` controls only the type-3 CONTROL literals such as `<|im_end|>`; the type-4
        USER_DEFINED ones such as `<think>` are matched either way.  See the note in `__init__`.
        `plain`: (start, end) spans of `text` that are ordinary text even where they spell a literal - a `</think>`
        or `<|user|>` quoted inside a message - and are tokenized with the text around them.
        """
        return self._encode_matching(text, self._special_re if parse_special else self._always_re, plain)

    def token_bytes(self, i: int) -> bytes:
        """The raw bytes of one token (a multi-byte character can be split across tokens)."""
        cache = self.__dict__.setdefault("_bytes_cache", {})
        b = cache.get(i)
        if b is None:
            if i < 0 or i >= len(self.tokens):
                raise IndexError("token id %d is outside the vocabulary (%d)" % (i, len(self.tokens)))
            alias = self.__dict__.get("aliases", {}).get(i)
            if alias is not None:
                b = cache[i] = alias.encode("utf-8")
                return b
            if self.token_types and self.token_types[i] in (3, 4):
                # a control / user-defined token is its text as is (llama.cpp's token_to_piece): GLM's are ASCII,
                # where the byte alphabet is the identity, DeepSeek's (<｜User｜>, ｜DSML｜) are not
                b = cache[i] = self.tokens[i].encode("utf-8")
                return b
            raw = bytearray()
            for ch in self.tokens[i]:
                v = UNICODE_TO_BYTE.get(ch)
                if v is None:
                    raise KeyError("token %d contains a character outside the byte alphabet: %r" % (i, ch))
                raw.append(v)
            b = cache[i] = bytes(raw)
        return b

    def decode(self, ids: list[int], errors: str = "replace") -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors=errors)


# ------------------------------------------------------------------ the pack's tokenizer/ directory
def tokenizer_files(md: dict) -> dict[str, str]:
    """{file name: text} of a pack's tokenizer/ folder from a GGUF's metadata (`md`: its tokenizer.ggml.* keys and
    tokenizer.chat_template, from any reader - tools/gguf_reader.py's GGUFFile.metadata, a pack tool's own kv).
    Written as UTF-8 text, chat_template.jinja with "\\n" line ends.  A pack tool calls this and writes the files into
    <pack>/tokenizer/ (maya-inkling2's: tools/inkling2/pack_gguf.py); extract() below does the same for one GGUF."""
    if md.get("tokenizer.ggml.model") != "gpt2":
        raise ValueError("expected a byte-level BPE tokenizer (model 'gpt2'), got %r" % md.get("tokenizer.ggml.model"))
    special = {k[len("tokenizer.ggml."):-len("_token_id")]: int(md[k]) for k in md
               if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")}
    tk = Tokenizer(list(md["tokenizer.ggml.tokens"]), list(md["tokenizer.ggml.merges"]),
                   [int(x) for x in md.get("tokenizer.ggml.token_type") or []] or None,
                   md.get("tokenizer.ggml.pre", "qwen35"), special)
    gpt4o = tk.pre in GPT4O_PRES
    cfg = {
        "model": "gpt2",
        "pre": tk.pre,
        "vocab_size": len(tk.tokens),
        "n_merges": len(tk.ranks),
        "special_ids": tk.special_ids,
        "add_bos_token": bool(md.get("tokenizer.ggml.add_bos_token", False)),
        # The pattern is SHIPPED, not recomputed by the reader: it is transcribed from llama.cpp for the
        # declared `pre` type, and a C++ port that re-derived it would be free to get `\p{M}` wrong again.
        "ignore_merges": tk.ignore_merges,
        "pre_pattern": GPT4O_LLAMA_PATTERN if gpt4o else GLM_PATTERN if tk.ignore_merges else QWEN35_PATTERN,
        "pre_pattern_source": (".ref/llama.cpp src/llama-vocab.cpp ~L455 (LLAMA_VOCAB_PRE_TYPE_GPT4O; run over the "
                               "text collapsed by unicode category, unicode.cpp unicode_regex_split)" if gpt4o else
                               ".ref/llama.cpp src/llama-vocab.cpp L406 (LLAMA_VOCAB_PRE_TYPE_CHATGLM4; "
                               "glm4/glm5 set ignore_merges)" if tk.ignore_merges else
                               ".ref/llama.cpp src/llama-vocab.cpp L396 (LLAMA_VOCAB_PRE_TYPE_QWEN35)"),
    }
    if gpt4o:
        cfg["pre_regex"] = "llama"   # ("unicode": GPT-4o's original regex instead - see GPT4O_COLLAPSED_PATTERN)
    if tk.aliases:
        cfg["aliases"] = {str(i): s for i, s in sorted(tk.aliases.items())}
    files = {"vocab.json": json.dumps(tk.ids, ensure_ascii=False),
             "merges.txt": "\n".join("%s %s" % k for k, _ in sorted(tk.ranks.items(), key=lambda kv: kv[1])),
             "token_type.json": json.dumps(tk.token_types),
             "tokenizer.json": json.dumps(cfg, ensure_ascii=False, indent=1)}
    # the model's own chat template: fine-tunes change it (Swift 1.5 differs from Qwen3.8-Flash-Next's)
    tpl = md.get("tokenizer.chat_template")
    if tpl:
        files["chat_template.jinja"] = tpl
    return files


def extract(gguf_path, out_dir) -> dict:
    """Write the tokenizer into `<out_dir>/tokenizer/` so the engine never opens the weight shards for it.  A gpt-4o /
    inkling2 GGUF (Inkling 2) gets tokenizer_files' folder (special_ids by their short names: bos, eos ...); every
    other one the folder as it always was (its special_ids by their GGUF keys, which the server reads as "no eos")."""
    out = pathlib.Path(out_dir) / "tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from gguf_reader import GGUFFile
    md = GGUFFile(pathlib.Path(gguf_path)).metadata
    if md.get("tokenizer.ggml.pre") in GPT4O_PRES:
        files = tokenizer_files(md)
        for name, text in files.items():
            (out / name).write_text(text, encoding="utf-8", newline="\n")
        return json.loads(files["tokenizer.json"])
    tk = Tokenizer.from_gguf(gguf_path)
    cfg = {
        "model": "gpt2",
        "pre": tk.pre,
        "vocab_size": len(tk.tokens),
        "n_merges": len(tk.ranks),
        "special_ids": tk.special_ids,
        "add_bos_token": False,
        # The pattern is SHIPPED, not recomputed by the reader: it is transcribed from llama.cpp for the
        # declared `pre` type, and a C++ port that re-derived it would be free to get `\p{M}` wrong again.
        "ignore_merges": tk.ignore_merges,
        "pre_pattern": GLM_PATTERN if tk.ignore_merges else QWEN35_PATTERN,
        "pre_pattern_source": (".ref/llama.cpp src/llama-vocab.cpp L406 (LLAMA_VOCAB_PRE_TYPE_CHATGLM4; "
                               "glm4/glm5 set ignore_merges)" if tk.ignore_merges else
                               ".ref/llama.cpp src/llama-vocab.cpp L396 (LLAMA_VOCAB_PRE_TYPE_QWEN35)"),
    }
    (out / "vocab.json").write_text(json.dumps(tk.ids, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join("%s %s" % k for k, _ in
                                              sorted(tk.ranks.items(), key=lambda kv: kv[1])), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(tk.token_types), encoding="utf-8")
    (out / "tokenizer.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    # the model's own chat template: fine-tunes change it (Swift 1.5 differs from Qwen3.8-Flash-Next's)
    tpl = md.get("tokenizer.chat_template")
    if tpl:
        (out / "chat_template.jinja").write_text(tpl, encoding="utf-8", newline="\n")
    return cfg


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="round-trip a corpus and report")
    args = ap.parse_args()
    cfg = extract(args.gguf, args.out)
    print("tokenizer/: vocab %d, merges %d, pre %s, specials %s"
          % (cfg["vocab_size"], cfg["n_merges"], cfg["pre"], cfg["special_ids"]))
    if args.check:
        tk = Tokenizer.from_gguf(args.gguf)
        corpus = ["", "Hello, world!", "  leading and trailing  ", "a\n\n\nb",
                  "def f(x):\n\treturn x  # comment\n", "\u4f60\u597d\uff0c\u4e16\u754c", "\u0645\u0631\u062d\u0628\u0627",
                  "\U0001f600\U0001f680\U0001f1fa\U0001f1f8", "e\u0301\u0301 combining", "\x00\x01\x7f control",
                  "\u00a0non-breaking\u00a0space", "1234567890", "MixedCASE_and-dashes", "\r\n\r\n", "\u2028\u2029",
                  "x" * 5000]
        bad = 0
        for s in corpus:
            ids = tk.encode(s)
            back = tk.decode(ids)
            if back != s:
                print("  *** ROUND TRIP FAILED *** %r -> %d ids -> %r" % (s[:40], len(ids), back[:40]))
                bad += 1
        print("round trip: %d strings, %d failed" % (len(corpus), bad))
        return 0 if bad == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
