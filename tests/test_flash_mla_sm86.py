import math
import random

import torch
import triton

from flash_mla import get_mla_metadata, flash_mla_with_kvcache


def scaled_dot_product_attention(query, key, value, h_q, h_kv, is_causal=False):
    query = query.float()
    key = key.float()
    value = value.float()
    key = key.repeat_interleave(h_q // h_kv, dim=0)
    value = value.repeat_interleave(h_q // h_kv, dim=0)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if is_causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weight += attn_bias
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight @ value, lse


def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    # print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    # # print nan coordinates of x
    # print("nan x", torch.isnan(x).nonzero(as_tuple=False))
    # print("nan y", torch.isnan(y).nonzero(as_tuple=False))
    assert cos_diff < 8e-5


@torch.inference_mode()
def test_flash_mla(b, s_q, mean_sk, h_q, h_kv, d, dv, causal, varlen):
    print(f"{b=}, {s_q=}, {mean_sk=}, {h_q=}, {h_kv=}, {d=}, {dv=}, {causal=}, {varlen=}")

    cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
    if varlen:
        for i in range(b):
            cache_seqlens[i] = max(random.normalvariate(mean_sk, mean_sk / 2), s_q)
    total_seqlens = cache_seqlens.sum().item()
    mean_seqlens = cache_seqlens.float().mean().int().item()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
    # print(f"{total_seqlens=}, {mean_seqlens=}, {max_seqlen=}")

    q = torch.randn(b, s_q, h_q, d)
    block_size = 32
    block_table = torch.arange(b * max_seqlen_pad // block_size, dtype=torch.int32).view(b, max_seqlen_pad // block_size)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
    for i in range(b):
        blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = float("nan")
    blocked_v = blocked_k[..., :dv]

    tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

    def flash_mla():
        return flash_mla_with_kvcache(
            q, blocked_k, block_table, cache_seqlens, dv,
            tile_scheduler_metadata, num_splits, causal=causal,
        )

    def ref_mla():
        out = torch.empty(b, s_q, h_q, dv, dtype=torch.float32)
        lse = torch.empty(b, h_q, s_q, dtype=torch.float32)
        for i in range(b):
            begin = i * max_seqlen_pad
            end = begin + cache_seqlens[i]
            O, LSE = scaled_dot_product_attention(
                q[i].transpose(0, 1),
                blocked_k.view(-1, h_kv, d)[begin:end].transpose(0, 1),
                blocked_v.view(-1, h_kv, dv)[begin:end].transpose(0, 1),
                h_q=h_q,
                h_kv=h_kv,
                is_causal=causal,
            )
            out[i] = O.transpose(0, 1)
            lse[i] = LSE
        return out, lse

    out_flash, lse_flash = flash_mla()
    out_torch, lse_torch = ref_mla()
    cal_diff(out_flash, out_torch, "out")
    cal_diff(lse_flash, lse_torch, "lse")

    t = triton.testing.do_bench(flash_mla)
    FLOPS = s_q * total_seqlens * h_q * (d + dv) * 2
    bytes = (total_seqlens * h_kv * d + b * s_q * h_q * d + b * s_q * h_q * dv) * (torch.finfo(dtype).bits // 8)
    print(f"{t:.3f} ms, {FLOPS / 10 ** 9 / t:.0f} TFLOPS, {bytes / 10 ** 6 / t:.0f} GB/s")


if __name__ == "__main__":
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)

    h_kv = 1
    # Use original dimensions to maintain compatibility with models like DeepSeek V3 and R1
    d, dv = 576, 512
    causal = True

    # Use extremely small batch sizes and sequence lengths for SM86 (RTX 30xx) to reduce memory usage
    # Consumer GPUs have very limited cache compared to datacenter GPUs
    # test_flash_mla(32, 2, 4096, 16, 1, d, dv, True, True)
    for b in [16]:  # Minimal batch size
        for s in [512, 1024]:  # Minimal sequence lengths
            for h_q in [8, 16]:  # Minimal number of heads
                for s_q in [1]:  # Only test MTP = 1 to reduce memory usage
                    for varlen in [False]:
                        test_flash_mla(b, s_q, s, h_q, h_kv, d, dv, causal, varlen)
