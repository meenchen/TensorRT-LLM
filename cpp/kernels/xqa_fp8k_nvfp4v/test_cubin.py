#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import ctypes
import math
from pathlib import Path

import torch
from cuda.bindings import driver


class KVCacheList(ctypes.Structure):
    _fields_ = [
        ("k_cache", ctypes.c_void_p),
        ("v_cache", ctypes.c_void_p),
        ("v_scale_cache", ctypes.c_void_p),
        ("page_table", ctypes.c_void_p),
        ("seq_lens", ctypes.c_void_p),
        ("max_pages_per_seq", ctypes.c_uint32),
    ]


def check(result):
    error, *values = result
    if error != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {error}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


class MixedXqaCubin:
    def __init__(self, path: Path):
        check(driver.cuInit(0))
        self.module = check(driver.cuModuleLoadData(bytearray(path.read_bytes())))
        self.function = check(driver.cuModuleGetFunction(self.module, b"kernel_mha"))

        smem_ptr, smem_bytes = check(driver.cuModuleGetGlobal(self.module, b"smemSize"))
        if smem_bytes != ctypes.sizeof(ctypes.c_uint32):
            raise RuntimeError(f"invalid smemSize symbol: {smem_bytes} bytes")
        smem_size = ctypes.c_uint32()
        check(driver.cuMemcpyDtoH(ctypes.addressof(smem_size), smem_ptr, ctypes.sizeof(smem_size)))
        self.smem_size = smem_size.value
        check(
            driver.cuFuncSetAttribute(
                self.function,
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                self.smem_size,
            )
        )

    def close(self):
        if self.module is not None:
            check(driver.cuModuleUnload(self.module))
            self.module = None

    def __call__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        v_scale_cache: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        output: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        semaphores: torch.Tensor,
        workspace: torch.Tensor,
    ):
        batch_size = q.shape[0]
        num_q_heads = q.shape[-2]
        num_kv_heads = k_cache.shape[-2]
        head_dim = q.shape[-1]
        page_size = k_cache.shape[1]
        max_seq_len = page_table.shape[-1] * page_size
        if (head_dim, num_q_heads // num_kv_heads, page_size) != (128, 4, 16):
            raise ValueError("this cubin is specialized for h128/g4/p16")

        cache_list = KVCacheList(
            k_cache.data_ptr(),
            v_cache.data_ptr(),
            v_scale_cache.data_ptr(),
            page_table.data_ptr(),
            seq_lens.data_ptr(),
            page_table.shape[-1],
        )

        k_head_containers = head_dim
        v_head_containers = head_dim // 2
        v_scale_head_containers = head_dim // 16

        def head_strides(tensor, containers):
            return tuple(tensor.stride(dim) // containers for dim in (0, 1, 2))

        k_strides = head_strides(k_cache, k_head_containers)
        v_strides = head_strides(v_cache, v_head_containers)
        v_scale_strides = head_strides(v_scale_cache, v_scale_head_containers)
        null = 0
        values = (
            num_kv_heads,
            1.0,
            null,
            output.data_ptr(),
            q.data_ptr(),
            null,
            cache_list,
            batch_size,
            1.0,
            k_scale.data_ptr(),
            1.0,
            v_scale.data_ptr(),
            *k_strides,
            *v_strides,
            0,
            0,
            0,
            *v_scale_strides,
            semaphores.data_ptr(),
            workspace.data_ptr(),
        )
        types = (
            ctypes.c_uint32,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            None,
            ctypes.c_uint32,
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_void_p,
            *(ctypes.c_uint32,) * 12,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
        subsequences = min(
            max(1, sm_count // (batch_size * num_kv_heads)),
            math.ceil(max_seq_len / 256),
        )
        stream = torch.cuda.current_stream(q.device).cuda_stream
        check(
            driver.cuLaunchKernel(
                self.function,
                subsequences,
                num_kv_heads,
                batch_size,
                128,
                1,
                2,
                self.smem_size,
                stream,
                (values, types),
                0,
            )
        )


def unpack_nvfp4(data: torch.Tensor, scales: torch.Tensor, global_scale):
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=data.device)
    nibbles = torch.stack((data & 0xF, data >> 4), dim=-1).flatten(-2)
    values = magnitudes[(nibbles & 7).long()]
    values = torch.where((nibbles & 8) != 0, -values, values)
    return values * scales.float().repeat_interleave(16, dim=-1) * global_scale


def reference(q, k_cache, v_cache, page_table, lengths, k_scale, v_scale):
    output = torch.empty_like(q)
    group_size = q.shape[-2] // k_cache.shape[-2]
    for batch_idx, seq_len in enumerate(lengths):
        pages = page_table[batch_idx]
        token_k = k_cache[pages].flatten(0, 1)[:seq_len]
        token_v = v_cache[pages].flatten(0, 1)[:seq_len]
        for q_head_idx in range(q.shape[-2]):
            kv_head_idx = q_head_idx // group_size
            scores = (token_k[:, kv_head_idx].float() @ q[batch_idx, 0, q_head_idx].float()) * (
                k_scale / math.sqrt(q.shape[-1])
            )
            output[batch_idx, 0, q_head_idx] = (
                torch.softmax(scores, dim=0)[:, None] * token_v[:, kv_head_idx].float()
            ).sum(dim=0)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cubin", type=Path)
    args = parser.parse_args()

    torch.manual_seed(7)
    torch.cuda.set_device(0)
    batch_size, page_size, num_kv_heads, num_q_heads, head_dim = 2, 16, 2, 8, 128
    pages_per_seq = 2
    num_pages = batch_size * pages_per_seq
    q = torch.randn(
        batch_size,
        1,
        num_q_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_scale = torch.tensor([0.025], dtype=torch.float32, device="cuda")
    v_scale = torch.tensor([0.125], dtype=torch.float32, device="cuda")
    k_cache = (
        (torch.randn(num_pages, page_size, num_kv_heads, head_dim, device="cuda") / k_scale)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    v_cache = torch.empty(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim // 2,
        dtype=torch.uint8,
        device="cuda",
    )
    v_cache[:, :, 0].fill_(0x77)
    v_cache[:, :, 1].fill_(0xFF)
    v_scale_cache = torch.ones(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim // 16,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    page_table = torch.arange(num_pages, dtype=torch.int32, device="cuda").view(
        batch_size, pages_per_seq
    )
    lengths = [17, 29]
    seq_lens = torch.tensor(lengths, dtype=torch.uint32, device="cuda")[:, None]
    output = torch.full_like(q, float("nan"))
    semaphores = torch.zeros(64, dtype=torch.uint32, device="cuda")
    workspace = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")

    launcher = MixedXqaCubin(args.cubin)
    try:
        launcher(
            q,
            k_cache,
            v_cache,
            v_scale_cache,
            page_table,
            seq_lens,
            output,
            k_scale,
            v_scale,
            semaphores,
            workspace,
        )
        torch.cuda.synchronize()
        v_dequant = unpack_nvfp4(v_cache, v_scale_cache, v_scale)
        expected = reference(q, k_cache, v_dequant, page_table, lengths, k_scale, v_scale)
        torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)
        max_abs = (output - expected).abs().max().item()
        print(f"PASS smem={launcher.smem_size} max_abs={max_abs:.6f}")
    finally:
        launcher.close()


if __name__ == "__main__":
    main()
