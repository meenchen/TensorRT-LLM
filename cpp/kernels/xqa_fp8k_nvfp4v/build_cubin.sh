#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <output.cubin> <head-dim> <q-heads-per-kv> <page-size>" >&2
  exit 2
fi

OUTPUT=$(realpath -m "$1")
HEAD_DIM=$2
HEAD_GROUP_SIZE=$3
PAGE_SIZE=$4
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

case "${PAGE_SIZE}" in
  16|32|64|128) ;;
  *) echo "page-size must be one of 16, 32, 64, 128" >&2; exit 2 ;;
esac

if (( HEAD_DIM < 16 || HEAD_DIM > 256 || HEAD_DIM % 16 != 0 )); then
  echo "head-dim must be a multiple of 16 in [16, 256]" >&2
  exit 2
fi

mkdir -p "$(dirname "${OUTPUT}")"
"${CUDA_HOME}/bin/nvcc" \
  -std=c++17 \
  -O3 \
  -cubin \
  -arch=sm_100a \
  --use_fast_math \
  --expt-relaxed-constexpr \
  -DNDEBUG=1 \
  -DGENERATE_CUBIN=1 \
  -DINPUT_FP16=0 \
  -DDTYPE=__nv_bfloat16 \
  -DBEAM_WIDTH=1 \
  -DUSE_INPUT_KV=0 \
  -DUSE_CUSTOM_BARRIER=1 \
  -DSLIDING_WINDOW=0 \
  -DLOW_PREC_OUTPUT=0 \
  -DSPEC_DEC=0 \
  -DCACHE_ELEM_ENUM=3 \
  -DK_CACHE_ELEM_ENUM=2 \
  -DV_CACHE_ELEM_ENUM=3 \
  -DHEAD_ELEMS="${HEAD_DIM}" \
  -DHEAD_GRP_SIZE="${HEAD_GROUP_SIZE}" \
  -DTOKENS_PER_PAGE="${PAGE_SIZE}" \
  -o "${OUTPUT}" \
  "$(dirname "$0")/mha.cu"
