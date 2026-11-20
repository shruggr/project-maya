import json
W = {"type": "function", "function": {"name": "get_weather", "description": "The weather of a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                    "required": ["city"]}}}
S = {"type": "function", "function": {"name": "search", "description": "Search the web: \"quoted\" & <tags> ünïcödé",
     "parameters": {"type": "object", "properties": {"q": {"type": "string"}, "n": {"type": "number", "default": 2.5},
                                                     "filters": {"type": "object"}}}}}
N = {"type": "function", "function": {"name": "noop"}}
U = [{"role": "user", "content": "Weather in Paris?"}]
def call(i, name, args):
    return {"id": i, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
render = [
 {"messages": U},
 {"messages": [{"role": "system", "content": "You are terse."}] + U},
 {"messages": [{"role": "developer", "content": "Dev note."}] + U},
 {"messages": U + [{"role": "assistant", "content": "Sunny."}, {"role": "user", "content": "And tomorrow?"}]},
 {"messages": U + [{"role": "assistant", "reasoning_content": "Think: Paris.", "content": "Sunny."},
                   {"role": "user", "content": "Thanks"}]},
 {"messages": U + [{"role": "assistant", "content": "", "tool_calls": [call("c1", "get_weather", {"city": "Paris", "days": 2})]},
                   {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 21}"},
                   {"role": "assistant", "content": "21 degrees."}, {"role": "user", "content": "ok"}], "tools": [W]},
 {"messages": U + [{"role": "assistant", "reasoning_content": "Two calls.", "content": "Checking.",
                    "tool_calls": [call("a", "get_weather", {"city": "Paris"}), call("b", "search", {"q": "Paris", "n": 3})]},
                   {"role": "tool", "tool_call_id": "a", "content": "sunny"},
                   {"role": "tool", "tool_call_id": "b", "content": "results"}], "tools": [W, S, N]},
 {"messages": U + [{"role": "assistant", "content": None, "tool_calls": [call("x", "search", {
     "q": "a \"b\" \\ c\nd ünï 日本 <tag>", "n": 1.5, "filters": {"z": [1, 2.0, -3e-7, 1e20, True, None], "a": {"é": 1, "e": 2, "B": 0}}})]},
                   {"role": "tool", "tool_call_id": "x", "content": "done"}], "tools": [S]},
 {"messages": [{"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]}]},
 {"messages": U + [{"role": "assistant", "content": ""}, {"role": "user", "content": "again"}]},
 {"messages": [{"role": "system", "content": "S1"}, {"role": "user", "content": "hi 👋 ÇA VA?"}], "tools": [N, W]},
]
for e in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
    render.append({"messages": U, "kwargs": {"reasoning_effort": e}, "effort": e})
H, E, T = "<|message_model|>", "<|end_message|>", "<|content_text|>"
parse = [
 H + "<|content_text|>Hello, world!\nWhat's up?" + E,
 H + "<|content_thinking|>I'm thinking" + E + H + T + "Hello!" + E,
 H + "<|content_thinking|>I was still thinking",
 H + "<|content_thinking|>a" + E + H + T + "Hello, wor",
 H + 'get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris","days":2}}' + E,
 H + "<|content_thinking|>call it" + E + H + T + "On it." + E + H + 'get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris"}}' + E,
 H + 'get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Rome"}}' + E + H + 'search<|content_invoke_tool_json|>{"name":"search","args":{"filters":{"a":[1,{"b":"}"}]},"n":1.5,"q":"x \\"y\\" \\\\ z\\n"}}' + E,
 H + 'search<|content_invoke_tool_json|>{"name":"search","args":{"q":"print(\'<|end_message|>\')"}}' + E,
 H + 'noop<|content_invoke_tool_json|>{"name":"noop","args":{}}' + E,
 H + "<|content_thinking|>one" + E + H + "<|content_thinking|>two" + E + H + T + "answer" + E,
]
cases = render + [{"parse": p, "messages": U, "tools": [W, S, N]} for p in parse]
json.dump(cases, open("cases.json", "w"), ensure_ascii=False)
print(len(render), len(parse))
