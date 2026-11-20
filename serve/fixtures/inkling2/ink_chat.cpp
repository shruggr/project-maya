// llama.cpp's own rendering and parsing of Inkling 2 chats (the reference for serve/test_inkling2_frontend.py):
//   ink_chat <template.jinja> <cases.json>  ->  one JSON line a case: {"prompt": ...} or {"parsed": {...}}
// a case: {"messages": [...], "tools": [...], "kwargs": {...}, "tool_choice": "auto|required|none"}  (render)
//         {"parse": "model output", "tools": [...], "parallel": bool}                                  (parse)
#include "chat.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <sstream>

using json = nlohmann::ordered_json;

static std::string slurp(const char * p) { std::ifstream f(p, std::ios::binary); std::stringstream s; s << f.rdbuf(); return s.str(); }

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: ink_chat template cases\n"); return 2; }
    auto tmpls = common_chat_templates_init(nullptr, slurp(argv[1]));
    json cases = json::parse(slurp(argv[2]));
    for (auto & c : cases) {
        json out;
        try {
            common_chat_templates_inputs in;
            in.messages = common_chat_msgs_parse_oaicompat(common_json::parse(c["messages"].dump()));
            if (c.contains("tools")) in.tools = common_chat_tools_parse_oaicompat(common_json::parse(c["tools"].dump()));
            std::string choice = c.value("tool_choice", "auto");
            in.tool_choice = choice == "required" ? COMMON_CHAT_TOOL_CHOICE_REQUIRED : choice == "none" ? COMMON_CHAT_TOOL_CHOICE_NONE : COMMON_CHAT_TOOL_CHOICE_AUTO;
            in.parallel_tool_calls = c.value("parallel", true);
            in.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
            in.add_generation_prompt = true;
            if (c.contains("kwargs")) for (auto & [k, v] : c["kwargs"].items()) in.chat_template_kwargs[k] = v.dump();
            if (c.contains("parse")) {
                common_chat_session_params sp;
                common_chat_session session(tmpls.get(), nullptr, in, sp);
                std::string text = c["parse"];
                const common_chat_msg & msg = session.finish(common_chat_input(text));
                json calls = json::array();
                for (auto & tc : msg.tool_calls) calls.push_back({{"name", tc.name}, {"arguments", tc.arguments}});
                out["parsed"] = {{"content", msg.content}, {"reasoning", msg.reasoning_content}, {"calls", calls}};
            } else {
                auto params = common_chat_templates_apply(tmpls.get(), in);
                out["prompt"] = params.prompt.str();
            }
        } catch (const std::exception & e) {
            out["error"] = e.what();
        }
        std::cout << out.dump() << "\n";
    }
    return 0;
}
