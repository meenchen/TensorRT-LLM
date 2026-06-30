# FP8-K/NVFP4-V XQA

This directory contains a public CUDA XQA decode specialization for a mixed
KV cache. K is FP8 E4M3. V is packed NVFP4 E2M1 with one E4M3 scale per 16
values. Only V is converted while loading the XQA shared-memory tile.

The checked-in cubin targets `sm_100a`, BF16 query/output, head dimension 128,
four query heads per KV head, and page size 16. Its SHA256 is:

```text
1250d87fa1268a44730f3ef80c8996e1b677a3d8222582b46951cddb3ca42e70
```

Rebuild and validate it with:

```bash
CUDA_HOME=/usr/local/cuda \
  ./build_cubin.sh /tmp/fp8_k_nvfp4_v_h128_g4_p16.cubin 128 4 16
python test_cubin.py /tmp/fp8_k_nvfp4_v_h128_g4_p16.cubin
```

The source is kept independently from `cpp/kernels/xqa` because the regular
TensorRT-LLM XQA ABI assumes one common K/V dtype. The runtime consumer must
pass independent K, packed V, and V-scale strides.

This is a native XQA cubin, not a TRTLLM-GEN FMHA specialization. The public
TRTLLM-GEN interface ships its generator in `libTrtLlmGenFmhaLib.a` and rejects
one-sided E2M1 K/V transforms before source generation. Adding the equivalent
tensor-core specialization requires that generator source; this directory does
not patch or bypass the prebuilt library.
