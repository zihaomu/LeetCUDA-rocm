"""Small, explicit tile comparison for the gfx1201 forward prototypes."""

import json

import torch
import triton
from torch.nn import functional as F

from leetcuda_rocm.environment import ROOT, verify
from leetcuda_rocm.matrix_attention import _attention_kernel, _gemm_kernel
from leetcuda_rocm.matrix_attention_check import compare


def main():
    torch.manual_seed(20261102)
    torch.set_grad_enabled(False)
    records = []
    identity = verify()
    dtype = torch.float16
    a = torch.randn(2048, 2048, device="cuda", dtype=dtype) * 0.5
    b = torch.randn_like(a)
    out = torch.empty_like(a)
    reference = a @ b
    for tm, tn, tk, warps, stages in (
        (64, 64, 32, 4, 2),
        (64, 64, 64, 4, 2),
        (64, 128, 32, 4, 2),
        (64, 128, 64, 4, 2),
        (128, 128, 32, 8, 2),
        (128, 128, 64, 8, 2),
    ):

        def launch():
            return _gemm_kernel[(triton.cdiv(2048, tm), triton.cdiv(2048, tn))](
                a,
                b,
                out,
                2048,
                2048,
                2048,
                *a.stride(),
                *b.stride(),
                TM=tm,
                TN=tn,
                TK=tk,
                num_warps=warps,
                num_stages=stages
            )

        kernel = launch()
        check = compare(out, reference, dtype)
        ms = triton.testing.do_bench_cudagraph(launch, rep=50)
        row = {
            "operator": "gemm",
            "config": [tm, tn, tk, warps, stages],
            "ms": ms,
            "registers": kernel.n_regs,
            "spills": kernel.n_spills,
            **check,
        }
        records.append(row)
        print(json.dumps(row), flush=True)
    for d in (64, 128):
        q, k, v = (
            torch.randn(1, 8, 1024, d, device="cuda", dtype=dtype) * 0.5
            for _ in range(3)
        )
        out = torch.empty_like(q)
        reference = F.scaled_dot_product_attention(q, k, v)
        for tq, tv, warps in (
            (32, 32, 4),
            (32, 64, 4),
            (64, 32, 4),
            (64, 64, 4),
            (128, 32, 8),
            (64, 64, 8),
        ):

            def launch():
                return _attention_kernel[(triton.cdiv(1024, tq), 8)](
                    q,
                    k,
                    v,
                    out,
                    8,
                    1024,
                    1024,
                    d,
                    *q.stride(),
                    *k.stride(),
                    *v.stride(),
                    SCALE=d**-0.5,
                    CAUSAL=False,
                    TQ=tq,
                    TV=tv,
                    TD=d,
                    num_warps=warps,
                    num_stages=1
                )

            kernel = launch()
            check = compare(out, reference, dtype, attention=True)
            ms = triton.testing.do_bench_cudagraph(launch, rep=50)
            row = {
                "operator": "attention",
                "d": d,
                "config": [tq, tv, warps],
                "ms": ms,
                "registers": kernel.n_regs,
                "spills": kernel.n_spills,
                **check,
            }
            records.append(row)
            print(json.dumps(row), flush=True)
    path = ROOT / "artifacts/rocm/matrix_attention/tile_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"environment": identity, "dtype": str(dtype), "records": records},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
