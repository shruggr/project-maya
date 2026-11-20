#!/bin/bash
set -e
L=/build/llama.cpp-inkling2
cd /work
/opt/rocm/llvm/bin/clang++ -std=c++17 -O1 -I$L/common -I$L/include -I$L/ggml/include -I$L/vendor -I$L/src ink_chat.cpp \
  -L$L/build-hip/bin -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,$L/build-hip/bin -o ink_chat
echo BUILT
