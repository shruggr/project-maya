#!/bin/bash
# usage: dspark_eval.sh <out name under /mnt/data/inkling2-ref> <N> <prompts "name=file,.."> <variants file>
OUT=$1; N=$2; PROMPTS=$3; VARS=$(cat "$4"); shift 4
mkdir -p /mnt/data/inkling2-ref/$OUT
exec podman run --rm --name inkling-dspark-eval --network=host --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined --ipc=host \
  -v /home/shruggr/build:/build:z -v /mnt/data/models/llm:/models:z -v /mnt/data/inkling2-ref:/ref:z \
  -e INK_OUT=/ref/$OUT -e INK_N=$N -e INK_PROMPTS="$PROMPTS" -e INK_VARIANTS="$VARS" -e INK_TAP_EXTRA=1 \
  --entrypoint /bin/bash localhost/vllm-rocm-ple-offload:latest -c "
export LD_LIBRARY_PATH=/build/llama.cpp-inkling2-dspark/build-hip/bin:\$LD_LIBRARY_PATH
/build/llama.cpp-inkling2-dspark/build-hip/bin/llama-inkling-dspark-eval \
  -m /models/Inkling-2-Small-GGUF/UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf \
  -md /models/Inkling-2-Small-MTP/Inkling-2-Small-DSpark-BF16.gguf --spec-type draft-dflash \
  -fa on -c 4096 -t 16 $*"
