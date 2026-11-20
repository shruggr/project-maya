"""serve/test_inkling2_frontend.py - Inkling 2 Small in the frontend and the server: Thinking Machines' template (the
GGUF's) detected and rendered as llama.cpp renders it, every client's spelling of a thinking level mapped onto the
template's own levels, the model's answers - typed blocks - read into reasoning, answer and tool calls as llama.cpp's
parser reads them, its stop tokens, a required call's opening, the loop guard's prompts, and the config's family.

    python -m unittest serve.test_inkling2_frontend

Offline, no GPU: the template is the copy in serve/fixtures/inkling2, llama.cpp's renders and parses of it are
serve/fixtures/inkling2/llama_chat_golden.json (made by llama.cpp's own chat code: ink_chat.cpp there), the tokenizer a
small stand-in with Inkling's special tokens as single tokens read back as the pack's tokenizer reads them (its
structural tokens as their marks), the engine a scripted one.
"""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import loop_guard, server
from serve.frontend import (INK_CALL, INK_END, INK_HEAD, INK_TEXT, INK_THINK, INKLING_MARKS, ChatTemplate, OutputParser,
                            anthropic_to_messages, effort_kwargs, inkling_marked, openai_to_messages)
from serve.server import ByteTokenizer, Service

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "serve" / "fixtures" / "inkling2"
TEMPLATE = FIX / "chat_template.jinja"
TEMPLATE_SHA256 = "d4c852bf15f2c08a1592d2024755f86427de75bc5e5e3272cd98da74ecb3475f"
GGUF = Path("/mnt/data/models/llm/Inkling-2-Small-GGUF/UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf")
H, E, TX, TH, TJ = ("<|message_model|>", "<|end_message|>", "<|content_text|>", "<|content_thinking|>",
                    "<|content_invoke_tool_json|>")
EOS = "<|content_model_end_sampling|>"
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "The weather of a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}]
ASK = [{"role": "user", "content": "Weather in Paris?"}]
CALL = H + "get_weather" + TJ + '{"name":"get_weather","args":{"city":"Paris","days":2}}' + E
PICTURE = {"role": "user", "content": [{"type": "text", "text": "What is this?"},
                                       {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}


def template() -> ChatTemplate:
    return ChatTemplate(TEMPLATE)


def golden() -> list:
    return json.loads((FIX / "llama_chat_golden.json").read_text(encoding="utf-8"))["cases"]


def level_line(prompt: str) -> str:
    """The level the prompt's effort block names (its end: the token's text, or its mark as the stand-in decodes it)."""
    return re.search("Thinking effort level: ([0-9.]+)(?:<|" + INK_END + ")", prompt).group(1)


class InkTok(ByteTokenizer):
    """ByteTokenizer with Inkling's special tokens as one token each, read back as the pack's tokenizer reads them: its
    structural tokens as their marks (tools/strata_tokenizer.py's aliases), the others as their text."""
    SPECIALS = ["<|endoftext|>", "<|message_user|>", H, "<|message_system|>", "<|message_tool|>", TX, "<|content_image|>",
                EOS, TH, E, "<|content_tool_error|>", "<|content_xml|>", TJ, "<|content_invoke_tool_text|>",
                "<|unused_200054|>"]
    special_tokens = {s: 256 + k for k, s in enumerate(SPECIALS)}
    special_ids = {"eos": special_tokens[EOS], "bos": special_tokens[EOS]}

    def decode(self, ids, errors="replace"):
        raw = bytearray()
        for t in ids:
            s = self.SPECIALS[t - 256] if t >= 256 else None
            raw += (INKLING_MARKS.get(s, s)).encode() if s is not None else bytes([t])
        return raw.decode("utf-8", errors=errors)


class Scripted:
    """Request k's answer is scripts[k] (the last one again after that): a string in the template's own tags (each tag
    the model's token), or a list of ids; then the turn's end.  Records every prompt and sampling.  Like maya-inkling2:
    no logprobs, no logit bias, one conversation."""

    def __init__(self, tok, *scripts):
        self.tok, self.max_context, self.scripts = tok, 32768, scripts
        self.prompts, self.sampling = [], []
        self.info = {"engine": "inkling2", "engine_kind": "inkling2"}

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        self.sampling.append(dict(sampling or {}))
        s = self.scripts[min(len(self.prompts), len(self.scripts)) - 1]
        ids = list(s) if isinstance(s, list) else self.tok.encode(s, parse_special=True)
        for t in (ids + [self.tok.special_tokens[EOS]])[:max_new]:
            yield t


def service(*scripts):
    tok = InkTok()
    eng = Scripted(tok, *scripts)
    return Service(eng, tok, template()), eng, tok


def answer(svc, req, anthropic=False):
    conv = anthropic_to_messages if anthropic else openai_to_messages
    msgs, tools, kw = conv(req, inkling=True)
    ids, thinking, max_new = svc.prepare(msgs, tools, kw, 500)
    sampling = {k: kw.get(k) for k in ("_prefill", "_blocked", "_think_resume")}
    events, done = [], None
    for kind, x in svc.run(ids, thinking, tools, max_new, sampling, threading.Event()):
        if kind == "event":
            events.append(x)
        elif kind == "done":
            done = x
    return {"content": "".join(e.text for e in events if e.kind == "content"),
            "reasoning": "".join(e.text for e in events if e.kind == "reasoning"),
            "calls": [(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"], "done": done}


def parse(text, step=3, marked=None, **kw):
    """The parser's events for a model answer written in the template's tags (each tag its token: its mark)."""
    p = OutputParser(inkling=True, stream_tools=kw.pop("stream_tools", True), **kw)
    t = marked if marked is not None else inkling_marked(text)
    ev = []
    for i in range(0, len(t), step):
        ev += p.feed(t[i:i + step])
    return ev + p.finish(), p


def summary(ev):
    return {"content": "".join(e.text for e in ev if e.kind == "content"),
            "reasoning": "".join(e.text for e in ev if e.kind == "reasoning"),
            "calls": [{"name": e.call.name, "arguments": e.call.arguments} for e in ev if e.kind == "tool_call"]}


class Template(unittest.TestCase):
    def test_fixture_is_the_ggufs(self):
        raw = TEMPLATE.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(raw).hexdigest(), TEMPLATE_SHA256)
        if GGUF.exists():
            import sys
            sys.path.insert(0, str(ROOT / "tools"))
            from gguf_reader import GGUFFile
            self.assertEqual(GGUFFile(GGUF).metadata["tokenizer.chat_template"], TEMPLATE.read_text(encoding="utf-8"))

    def test_detected(self):
        t = template()
        self.assertTrue(t.inkling)
        self.assertFalse(t.hy4 or t.minimax or t.deepseek or t.qwen or t.glm_effort or t.glm_dsa or t.forced_thinking)
        for path in ("serve/chat_template.jinja", "serve/glm5_chat_template.jinja", "tools/mm3/chat_template.jinja",
                     "serve/fixtures/hy4_flash/chat_template.jinja"):
            if (ROOT / path).exists():
                self.assertFalse(ChatTemplate(ROOT / path).inkling, path)

    def test_renders_as_llama_cpp(self):
        """Every conversation as llama.cpp's chat code renders it (common_chat_templates_apply) - but for an assistant
        turn with nothing in it, which the server never renders (openai_to_messages: a history of them teaches the
        model to answer with nothing)."""
        t, n = template(), 0
        for c in golden():
            case, want = c["case"], c["llama"].get("prompt")
            if "parse" in case:
                continue
            req = {"messages": case["messages"], "tools": case.get("tools")}
            if "effort" in case:
                req["reasoning_effort"] = case["effort"]
            msgs, tools, kw = openai_to_messages(req, inkling=True)
            got = t.render(msgs, tools, **{k: v for k, v in kw.items() if not k.startswith("_")})
            empty = [m for m in case["messages"] if m["role"] == "assistant" and not m.get("content")
                     and not m.get("tool_calls") and not m.get("reasoning_content")]
            if empty:
                self.assertEqual(got, want.replace(H + TX + E + EOS, ""))
            else:
                self.assertEqual(got, want, json.dumps(case)[:120])
            n += 1
        self.assertEqual(n, 18)

    def test_levels(self):
        t = template()
        for kwargs, want in (({}, "0.9"), ({"reasoning_effort": "auto"}, "0.9"), ({"enable_thinking": False}, "0"),
                             ({"reasoning_effort": "low"}, "0.2"), ({"reasoning_effort": "high"}, "0.9"),
                             ({"reasoning_effort": "max"}, "0.99"),
                             ({"reasoning_effort": "high", "effort_name": "medium"}, "0.7"),
                             ({"reasoning_effort": "max", "effort_name": "xhigh"}, "0.99")):
            self.assertEqual(level_line(t.render(ASK, **kwargs)), want, kwargs)
        for word, want in (("none", "0"), ("off", "0"), ("minimal", "0.1"), ("disabled", "0"), ("low", "0.2"),
                           ("medium", "0.7"), ("high", "0.9"), ("xhigh", "0.99"), ("max", "0.99"),
                           ("maximum", "0.99"), ("auto", "0.9"), ("default", "0.9"), (False, "0")):
            self.assertEqual(level_line(t.render(ASK, **effort_kwargs(word))), want, word)

    def test_levels_shown(self):
        t = template()
        self.assertEqual(t.thinking_levels(), ["none", "low", "medium", "high", "xhigh", "max"])
        self.assertEqual([t.effort_shown(w) for w in ("none", "minimal", "low", "medium", "high", "xhigh", "max",
                                                      "auto", "bogus")],
                         ["none", "minimal", "low", "medium", "high", "xhigh", "max", "high", None])
        self.assertIsNone(ChatTemplate(ROOT / "serve/chat_template.jinja").effort_shown("high"))

    def test_requests(self):
        def level(req, anthropic=False):
            msgs, tools, kw = (anthropic_to_messages if anthropic else openai_to_messages)(req, inkling=True)
            return level_line(template().render(msgs, tools, **{k: v for k, v in kw.items() if not k.startswith("_")}))
        for req, want in (({}, "0.9"), ({"reasoning_effort": "minimal"}, "0.1"), ({"reasoning": {"effort": "low"}}, "0.2"),
                          ({"chat_template_kwargs": {"reasoning_effort": "medium"}}, "0.7"),
                          ({"chat_template_kwargs": {"enable_thinking": False}}, "0")):
            self.assertEqual(level({"messages": ASK, **req}), want, req)
        base = {"messages": ASK, "max_tokens": 100}
        for req, want in (({"thinking": {"type": "disabled"}}, "0"),
                          ({"thinking": {"type": "enabled", "budget_tokens": 1024}}, "0.2"),
                          ({"thinking": {"type": "enabled", "budget_tokens": 4096}}, "0.9"),
                          ({"thinking": {"type": "enabled", "budget_tokens": 32000}}, "0.99"),
                          ({"output_config": {"effort": "medium"}}, "0.7")):
            self.assertEqual(level({**base, **req}, anthropic=True), want, req)
        with self.assertRaises(ValueError):
            openai_to_messages({"messages": ASK, "reasoning_effort": "enormous"}, inkling=True)

    def test_tool_results_named_by_their_call(self):
        """A tool result's block is headed by its tool's name - OpenAI's by its tool_call_id's call (llama.cpp), and
        Anthropic's tool_result blocks the same by their place."""
        t = template()
        req = {"model": "x", "max_tokens": 50, "tools": [{"name": "get_weather", "input_schema": TOOLS[0]["function"]["parameters"]},
                                                         {"name": "search", "input_schema": {"type": "object"}}],
               "messages": [{"role": "user", "content": "Paris?"},
                            {"role": "assistant", "content": [
                                {"type": "thinking", "thinking": "Two."},
                                {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}},
                                {"type": "tool_use", "id": "t2", "name": "search", "input": {"q": "Paris"}}]},
                            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "sunny"},
                                                         {"type": "tool_result", "tool_use_id": "t2", "content": "found"},
                                                         {"type": "text", "text": "and?"}]}]}
        msgs, tools, kw = anthropic_to_messages(req, inkling=True)
        out = t.render(msgs, tools)
        self.assertIn("<|message_tool|>get_weather" + TX + "sunny" + E + "<|message_tool|>search" + TX + "found" + E, out)
        self.assertIn(H + TH + "Two." + E + H + "get_weather" + TJ + '{"name":"get_weather","args":{"city":"Paris"}}' + E,
                      out)
        self.assertTrue(out.endswith("<|message_user|>" + TX + "and?" + E))

    def test_text_parts_stay_blocks(self):
        msgs, _, _ = openai_to_messages({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}]}, inkling=True)
        self.assertEqual(msgs[0]["content"], [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}])
        msgs, _, _ = openai_to_messages({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}]})
        self.assertEqual(msgs[0]["content"], "onetwo")                  # (every other model: as before)

    def test_prompt_opens_nothing(self):
        t = template()
        out = t.render(ASK)
        self.assertTrue(out.endswith("<|message_user|>" + TX + "Weather in Paris?" + E))
        self.assertFalse(t.opens_thinking(out))
        self.assertEqual(t.inkling_call_open("get_weather"), H + "get_weather" + TJ + '{"name":"get_weather","args":')
        self.assertEqual(t.inkling_call_open(), H)


class Parser(unittest.TestCase):
    def test_parses_as_llama_cpp(self):
        """Every answer as llama.cpp's Inkling parser reads it (common_chat_session), whatever the deltas."""
        n = 0
        for c in golden():
            case = c["case"]
            if "parse" not in case:
                continue
            want = dict(c["llama"]["parsed"])
            want["calls"] = [{"name": x["name"], "arguments": json.loads(x["arguments"])} for x in want["calls"]]
            # (the model's tokens: the structure's are its special tokens; text inside a JSON string is text)
            marked = inkling_marked(case["parse"]).replace("'" + INK_END + "'", "'<|end_message|>'")
            for step in (1, 2, 5, 1000):
                ev, _ = parse(None, step, marked=marked)
                self.assertEqual(summary(ev), want, (case["parse"][:60], step))
            n += 1
        self.assertEqual(n, 10)

    def test_tags_in_text_are_text(self):
        ev, _ = parse(H + TH + "About </think> and <tool_call>x</tool_call>." + E + H + TX +
                      "Use <tool_call>, </think> or \x00think\x00 as text." + E)
        self.assertEqual(summary(ev), {"reasoning": "About </think> and <tool_call>x</tool_call>.",
                                       "content": "Use <tool_call>, </think> or \x00think\x00 as text.", "calls": []})
        # (a structural token's text written in ordinary pieces stays text: only the token is its mark)
        ev, _ = parse(None, marked=INK_HEAD + INK_TEXT + "The block ends with <|end_message|>." + INK_END)
        self.assertEqual(summary(ev)["content"], "The block ends with <|end_message|>.")

    def test_streams_the_call(self):
        ev, _ = parse(H + TH + "Look." + E + CALL, step=1)
        kinds = [e.kind for e in ev]
        self.assertLess(kinds.index("tool_start"), kinds.index("tool_call"))
        args = "".join(e.text for e in ev if e.kind == "tool_args")
        self.assertEqual(json.loads(args), {"city": "Paris", "days": 2})
        self.assertGreater(kinds.count("tool_args"), 3)                # (the string argument piece by piece)
        call = [e.call for e in ev if e.kind == "tool_call"][0]
        self.assertEqual((call.name, call.arguments), ("get_weather", {"city": "Paris", "days": 2}))
        self.assertEqual(call.id, [e.call for e in ev if e.kind == "tool_start"][0].id)

    def test_other_json_forms_read_whole(self):
        ev, _ = parse(H + "get_weather" + TJ + ' { "args" : {"city": "Rome"}, "name": "get_weather" } ' + E)
        self.assertEqual(summary(ev)["calls"], [{"name": "get_weather", "arguments": {"city": "Rome"}}])
        ev, _ = parse(H + "<|content_invoke_tool_text|>" + '{"name":"get_weather","args":{"city":"Oslo"}}' + E)
        self.assertEqual(summary(ev)["calls"], [{"name": "get_weather", "arguments": {"city": "Oslo"}}])

    def test_broken_calls(self):
        ev, p = parse(H + "get_weather" + TJ + "not json" + E + H + TX + "Sorry." + E)
        s = summary(ev)
        self.assertEqual((s["calls"], s["content"]), ([], "not jsonSorry."))
        self.assertEqual(p.malformed, ["not json"])
        # cut off by max_tokens inside an announced call: incomplete (strict), or its JSON closed (not strict)
        for strict in (True, False):
            ev, _ = parse(H + "get_weather" + TJ + '{"name":"get_weather","args":{"city":"Pa', strict_tools=strict)
            kinds = [e.kind for e in ev]
            self.assertIn("tool_start", kinds)
            if strict:
                self.assertTrue(any(e.kind in ("tool_call", "tool_incomplete") and not e.complete for e in ev))
            else:
                self.assertEqual(json.loads("".join(e.text for e in ev if e.kind == "tool_args")), {"city": "Pa"})
        # not streamed: the call's text as the model wrote it
        ev, _ = parse(H + "get_weather" + TJ + '{"name":"get_weather","args":{"ci', stream_tools=False)
        self.assertEqual(summary(ev), {"content": '{"name":"get_weather","args":{"ci', "reasoning": "", "calls": []})

    def test_outside_blocks_and_single_tool(self):
        ev, _ = parse("Hello without blocks.")
        self.assertEqual(summary(ev)["content"], "Hello without blocks.")
        ev, _ = parse(" \n" + H + TX + "Hi." + E + "\n")
        self.assertEqual(summary(ev)["content"], "Hi.")
        ev, p = parse(CALL + H + "get_weather" + TJ + '{"name":"get_weather","args":{"city":"Rome"}}' + E, single_tool=True)
        self.assertEqual(len(summary(ev)["calls"]), 1)
        self.assertTrue(p.finished)

    def test_state_follows_the_blocks(self):
        p = OutputParser(thinking=True, inkling=True, stream_tools=True)
        self.assertEqual(p.state, "content")                             # (the prompt opens no block)
        p.feed(INK_HEAD + INK_THINK + "hm")
        self.assertEqual(p.state, "reasoning")
        p.feed(INK_END + INK_HEAD + "get_weather" + INK_CALL + '{"name":"get_weather","args":{')
        self.assertTrue(p.call_opened)
        p.feed('"city":"x"}}' + INK_END)
        self.assertEqual(p.state, "content")


class Tokens(unittest.TestCase):
    def test_stops_and_no_think_end(self):
        svc, _, tok = service("x")
        ids = tok.special_tokens
        self.assertLessEqual({ids[EOS], ids["<|endoftext|>"], ids["<|message_user|>"], ids["<|message_system|>"],
                              ids["<|message_tool|>"]}, svc.stop_ids)
        self.assertNotIn(ids[E], svc.stop_ids)
        self.assertEqual(svc.eos_ids, {ids[EOS]})
        self.assertIsNone(svc.think_end_id)                              # (no thinking budget: no end-of-thinking token)
        self.assertFalse(svc.think_budget_info()["supported"])

    def test_loop_guard_sees_the_calls(self):
        self.assertTrue(loop_guard._TOOL_OPEN.search("x" + INK_CALL))
        self.assertTrue(loop_guard._TOOL_CLOSE.search("}" + INK_END))
        self.assertIsNone(loop_guard._TOOL_OPEN.search("<|content_invoke_tool_json|>"))   # (its text: not the token)

    def test_marks_are_the_tokenizers(self):
        import sys
        sys.path.insert(0, str(ROOT / "tools"))
        import strata_tokenizer
        self.assertEqual(strata_tokenizer.INKLING2_MARKS, INKLING_MARKS)


class Serving(unittest.TestCase):
    def test_thinking_and_answer(self):
        svc, eng, tok = service(H + TH + "Thinking." + E + H + TX + "Hello." + E)
        got = answer(svc, {"messages": ASK})
        self.assertEqual((got["reasoning"], got["content"], got["calls"]), ("Thinking.", "Hello.", []))
        self.assertEqual(got["done"]["finish"], "stop")
        prompt = tok.decode(eng.prompts[0])
        self.assertTrue(prompt.endswith("Weather in Paris?" + INKLING_MARKS[E]))   # (no generation prefix)
        self.assertIn("Thinking effort level: 0.9", prompt)

    def test_tool_call(self):
        svc, eng, tok = service(H + TH + "Look it up." + E + CALL)
        got = answer(svc, {"messages": ASK, "tools": TOOLS})
        self.assertEqual(got["calls"], [("get_weather", {"city": "Paris", "days": 2})])
        self.assertEqual(got["reasoning"], "Look it up.")
        self.assertIn("tool_declare<|content_xml|>", bytes(t for t in eng.prompts[0] if t < 256).decode() +
                      tok.decode(eng.prompts[0]))

    def test_required_call_opens_in_the_models_own_tags(self):
        svc, eng, tok = service('{"city":"Rome"}}' + E)
        got = answer(svc, {"messages": ASK, "tools": TOOLS,
                           "tool_choice": {"type": "function", "function": {"name": "get_weather"}}})
        want = tok.encode(H + "get_weather" + TJ + '{"name":"get_weather","args":', parse_special=True)
        self.assertEqual(eng.prompts[0][-len(want):], want)
        self.assertEqual(got["calls"], [("get_weather", {"city": "Rome"})])
        self.assertEqual(level_line(tok.decode(eng.prompts[0])), "0")   # (a forced call: thinking off)
        # required, no name: the block's header alone, the model names the tool
        svc, eng, tok = service('get_weather' + TJ + '{"name":"get_weather","args":{"city":"Oslo"}}' + E)
        got = answer(svc, {"messages": ASK, "tools": TOOLS, "tool_choice": "required"})
        self.assertEqual(eng.prompts[0][-1], tok.special_tokens[H])
        self.assertEqual(got["calls"], [("get_weather", {"city": "Oslo"})])

    def test_levels_reach_the_prompt(self):
        svc, eng, tok = service(H + TX + "ok" + E)
        svc.default_effort = "low"
        answer(svc, {"messages": ASK})
        answer(svc, {"messages": ASK, "reasoning_effort": "xhigh"})
        answer(svc, {"messages": ASK, "chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual([level_line(tok.decode(p)) for p in eng.prompts], ["0.2", "0.99", "0"])

    def test_quoted_tags_are_text_in_the_prompt(self):
        svc, eng, tok = service(H + TX + "ok" + E)
        answer(svc, {"messages": [{"role": "user", "content": "Quote " + E + " and " + EOS + " please"}]})
        prompt = eng.prompts[0]
        self.assertEqual(prompt.count(tok.special_tokens[EOS]), 0)       # (the user's text: plain bytes)
        self.assertIn(("Quote " + E + " and " + EOS + " please").encode(), bytes(t for t in prompt if t < 256))

    def test_pictures_are_refused_in_plain_words(self):
        svc, eng, _ = service("x")
        self.assertEqual(svc.text_only(), server.INKLING_TEXT_ONLY)
        msgs, tools, kw = openai_to_messages({"messages": [PICTURE]}, inkling=True)
        with self.assertRaises(ValueError) as e:
            svc.prepare(msgs, tools, kw, 100)
        self.assertIn("reads text only", str(e.exception))
        self.assertEqual(eng.prompts, [])
        req = {"messages": ASK + [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Rome\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": [
                {"type": "text", "text": "a map:"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,BB"}}]}],
            "tools": TOOLS}
        msgs, tools, kw = openai_to_messages(req, inkling=True)
        svc.prepare(msgs, tools, kw, 100)                                 # (a tool result's picture: a note, no error)

    def test_loop_guard_suffixes(self):
        svc, _, tok = service("x")
        flat = [t["function"] for t in TOOLS]
        ids, thinks = svc._keep_suffix(flat, {})
        self.assertFalse(thinks)
        self.assertEqual(ids[0], tok.special_tokens[EOS])               # (after the model's own <|end_message|>)
        self.assertEqual(ids[1], tok.special_tokens["<|message_user|>"])
        key = ("get_weather", loop_guard.args_key({"city": "Paris"}))
        out = svc._block_suffixes({key: loop_guard.ToolRepeat("get_weather", 3)}, flat, {})
        self.assertIsNotNone(out)
        _, ids, thinks = out[key]
        self.assertFalse(thinks)
        self.assertEqual(ids[:2], [tok.special_tokens[EOS], tok.special_tokens["<|message_tool|>"]])
        self.assertIn(b"get_weather", bytes(t for t in ids if t < 256))

    def test_detokenize_names_the_tokens(self):
        svc, _, tok = service("x")
        ids = tok.encode(H + TX + "hi" + E + EOS, parse_special=True)
        self.assertEqual(svc.detokenize({"tokens": ids})["content"], H + TX + "hi" + E + EOS)

    def test_logprobs_and_bias_are_refused(self):
        svc, _, _ = service("x")
        with self.assertRaises(ValueError):
            svc.check_request({"messages": ASK, "logprobs": True})
        with self.assertRaises(ValueError):
            svc.check_request({"messages": ASK, "logit_bias": {"65": 5}})


class Config(unittest.TestCase):
    """A config that runs maya-inkling2 (docs/INKLING2_SERVING.md): its family everywhere, its defaults, its command."""
    CFG = {"exe": "/home/me/project-maya/build-hip/maya-inkling2",
           "args": ["--pack", "/models/Inkling-2-Small-GGUF/UD-Q2_K_XL/maya-inkling2", "--max-context", "32768"],
           "tokenizer": "/models/Inkling-2-Small-GGUF/UD-Q2_K_XL/maya-inkling2/tokenizer",
           "model_name": "inkling-2-small", "gpu": [0, 1, 2, 3], "backend": "hip",
           "sampling": {"temperature": 1.0, "top_p": 1.0}, "reasoning_effort": "high",
           "installer": {"family": "inkling2", "inkling2_quant": "ud-q2_k_xl"}}

    def test_family_and_defaults(self):
        Q = server.maya_quants
        self.assertEqual(Q.family_of(self.CFG), "inkling2")
        bare = {k: v for k, v in self.CFG.items() if k != "installer"}   # (by its engine's name alone: not DeepSeek's)
        self.assertEqual(Q.family_of(bare), "inkling2")
        self.assertEqual(Q.model_defaults(self.CFG), ({"temperature": 1.0, "top_p": 1.0}, "high"))
        self.assertEqual(Q.config_label(self.CFG), "Inkling 2 Small · UD-Q2_K_XL")
        self.assertEqual(Q.config_stem("inkling2", "ud-q2_k_xl"), "maya-inkling2-small")
        cfg = server.checked_config_effort({k: v for k, v in self.CFG.items() if k != "reasoning_effort"}, template(), True)
        self.assertEqual(cfg["reasoning_effort"], "high")

    def test_engine_line(self):
        args = server.engine_args(self.CFG)
        self.assertEqual(args[:4], ["--pack", self.CFG["args"][1], "--max-context", "32768"])
        self.assertEqual(args[4:], ["--layer-split", "auto"])            # (several GPUs: the server's layer split)
        self.assertEqual(server.child_env(self.CFG)["HIP_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertEqual(server.StrataEngine.sampling_keys({"temperature": 1.0, "top_p": 1.0}), " temperature=1.0")

    def test_models_page(self):
        from serve import models
        self.assertEqual(models.family(self.CFG), "inkling2")
        s = models.settings_of(self.CFG)
        self.assertEqual((s["context"], s["gpus"], s["images"]), (32768, [0, 1, 2, 3], False))
        self.assertEqual(models.choices(self.CFG)["images"], False)
        out = models.apply(self.CFG, {"context": 65536, "gpus": [0, 1]})
        self.assertEqual(out["args"], ["--pack", self.CFG["args"][1], "--max-context", "65536"])
        self.assertEqual(out["gpu"], [0, 1])
        with self.assertRaises(ValueError):
            models.apply(self.CFG, {"expert_cache": 100})                 # (GLM's: not this engine's)
        with self.assertRaises(ValueError):
            models.apply(self.CFG, {"images": True})

    def test_setup_side(self):
        import maya
        self.assertEqual(maya.config_family(self.CFG), "inkling2")
        self.assertEqual(maya.setup_title(self.CFG), "Inkling-2-Small")
        self.assertNotIn("inkling2", maya.FAMILIES)                       # (no setup flow for it yet)
        maya.select_build_backend("hip")
        self.assertEqual(maya.exe_for_target("maya-inkling2").name, "maya-inkling2")
        self.assertIsNone(maya.mixed_family(self.CFG))
        self.assertFalse(maya.needs_speed_calibration(self.CFG))
        self.assertFalse(maya.needs_prefill_calibration(self.CFG))

    def test_server_command(self):
        import maya
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            exe, pack = d / "build-hip" / "maya-inkling2", d / "maya-inkling2"
            exe.parent.mkdir()
            exe.write_text("")
            (pack / "tokenizer").mkdir(parents=True)
            (pack / "tokenizer" / "vocab.json").write_text("{}")
            cfg = dict(self.CFG, exe=str(exe), args=["--pack", str(pack), "--max-context", "32768"],
                       tokenizer=str(pack / "tokenizer"), port=8099)
            path = d / "maya-inkling2-small.json"
            path.write_text(json.dumps(cfg))
            a = type("A", (), {"port": None, "host": None, "api_key": None, "gpus": None, "gpu": None, "parallel": None,
                               "yes": True})()
            cmd, env, info = maya.server_command(path, a)
            self.assertEqual(cmd[1:7], [str(ROOT / "serve" / "server.py"), "--engine", "maya", "--config", str(path),
                                        "--port"])
            self.assertEqual(cmd[7], "8099")
            self.assertEqual(json.loads(path.read_text())["args"], cfg["args"])   # (nothing of another engine added)


class Http(unittest.TestCase):
    """What the page and a client see: /health's levels, a chat at every level, the Anthropic and Responses APIs."""

    @classmethod
    def setUpClass(cls):
        cls.svc, cls.eng, _ = service(H + TH + "Thinking." + E + H + TX + "Hello." + E)
        cls.svc.default_effort = "high"
        cls.httpd = server.serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def call(self, path, body=None):
        req = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    def test_health(self):
        code, h = self.call("/health")
        self.assertEqual(code, 200)
        self.assertEqual(h["thinking_levels"], ["none", "low", "medium", "high", "xhigh", "max"])
        self.assertEqual(h["defaults"]["reasoning_effort"], "high")
        self.assertEqual((h["images"], h.get("text_only")), (False, True))

    def test_chat_at_every_level(self):
        for level in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            code, out = self.call("/v1/chat/completions", {"messages": ASK, "reasoning_effort": level})
            self.assertEqual(code, 200, (level, out))
        code, out = self.call("/v1/chat/completions", {"messages": ASK})
        msg = out["choices"][0]["message"]
        self.assertEqual((msg.get("reasoning_content"), msg["content"]), ("Thinking.", "Hello."))
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")

    def test_messages_and_responses(self):
        code, out = self.call("/v1/messages", {"model": "inkling-2-small", "max_tokens": 200, "messages": ASK,
                                               "thinking": {"type": "enabled", "budget_tokens": 1024}})
        self.assertEqual(code, 200, out)
        self.assertEqual([b["type"] for b in out["content"]], ["thinking", "text"])
        code, out = self.call("/v1/responses", {"model": "inkling-2-small", "input": "Hi", "reasoning": {"effort": "low"}})
        self.assertEqual(code, 200, out)

    def test_a_picture_is_a_400(self):
        code, out = self.call("/v1/chat/completions", {"messages": [PICTURE]})
        self.assertEqual(code, 400)
        self.assertIn("reads text only", json.dumps(out))


if __name__ == "__main__":
    unittest.main()
