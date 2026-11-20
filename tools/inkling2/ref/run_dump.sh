#!/bin/bash
# Reference tensor dump for maya-inkling2 (docs/INKLING2_PORT.md): llama.cpp's Inkling 2 (checkout
# /home/shruggr/build/llama.cpp-inkling2 with examples/inkling-dump = this directory's inkling-dump.cpp), run in the
# ROCm container on this PC's GPUs.
#   run_dump.sh <name> <prompt file> [steps] [regex]   ->  /mnt/data/inkling2-ref/<name>/
#   LLAMA_ARGS replaces the attention flags (default "-fa on"), e.g. LLAMA_ARGS="-fa off -ctk f32 -ctv f32".
set -euo pipefail
NAME=$1; PF=$(realpath "$2"); STEPS=${3:-4}
FILTER=${4:-'^(inp_embd|attn_out-|ffn_inp-|ffn_out-|l_out-|ffn_moe_topk-|ffn_moe_weights|ffn_moe_probs-|result_norm|result_output|(Qcur|Kcur|Vcur|Rcur|kq_b|kqv_out)-(0|5)$)'}
mkdir -p /mnt/data/inkling2-ref/$NAME
exec podman run --rm --name inkling-dump-$NAME --network=host --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined --ipc=host \
  -v /home/shruggr/build:/build:z -v /mnt/data/models/llm:/models:z -v /mnt/data/inkling2-ref:/ref:z -v "$(dirname "$PF")":/prompt:z \
  -e INK_DUMP_DIR=/ref/$NAME -e INK_DUMP_STEPS=$STEPS -e INK_DUMP_FILTER="$FILTER" \
  --entrypoint /bin/bash localhost/vllm-rocm-ple-offload:latest -c "
export LD_LIBRARY_PATH=/build/llama.cpp-inkling2/build-hip/bin:\$LD_LIBRARY_PATH
/build/llama.cpp-inkling2/build-hip/bin/llama-inkling-dump \
  -m /models/Inkling-2-Small-GGUF/UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf \
  ${LLAMA_ARGS:--fa on} -c 4096 -b 4096 -ub 4096 -t 16 -f /prompt/$(basename "$PF")"
