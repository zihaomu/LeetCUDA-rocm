"""Independent numerical gates for the second batch of original HIP kernels."""

import torch


def gemm_checks(module, lib):
    names = sorted(
        name
        for name in dir(lib)
        if (name.startswith("hgemm_") and "cublas" not in name)
        or (module == "sgemm" and "_sliced_k16_" in name)
    )
    dtype = torch.float16 if module == "hgemm" else torch.float32
    # Keep the established HGEMM gate. AMD WMMA explicitly uses FP32 state.
    atol, rtol = (0.01, 0.01) if dtype == torch.float16 else (2e-4, 2e-4)
    results = []
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for name in names:
            bk = (
                32
                if "_k32_" in name or name == "hgemm_sliced_k_f16"
                else 16 if "_k16_" in name else 8
            )
            for tiles in (1, 2, 3, 5, 8):
                m, n, k = 128, 256, bk * tiles
                a = torch.randn(m, k, device="cuda", dtype=dtype) * 0.25
                b = torch.randn(k, n, device="cuda", dtype=dtype) * 0.25
                out = torch.full(
                    (m, n), float("nan"), device="cuda", dtype=dtype
                )
                options = (3, True, 256) if "_stages" in name else ()
                getattr(lib, name)(a, b, out, *options)
                expected = a.double() @ b.double()
                failure = None
                try:
                    torch.testing.assert_close(
                        out, expected.to(dtype), atol=atol, rtol=rtol
                    )
                except AssertionError as exc:
                    failure = str(exc)
                    print(
                        f"[{name}, K={k}, tiles={tiles}] FAIL: {failure}",
                        flush=True,
                    )
                error = out.double() - expected
                results.append(
                    {
                        "function": name,
                        "shape_mnk": [m, n, k],
                        "k_tiles": tiles,
                        "dtype": str(dtype),
                        "reference": "FP64 matmul, rounded for assert_close",
                        "max_abs_error": error.abs().max().item(),
                        "relative_l2_error": (
                            error.norm() / expected.norm()
                        ).item(),
                        "atol": atol,
                        "rtol": rtol,
                        "stream": "non-default",
                        "passed": failure is None,
                        **({"error": failure} if failure else {}),
                    }
                )
    stream.synchronize()
    return results


def merge_reference(prefix, prefix_lse, suffix, suffix_lse):
    # Use FP64 logaddexp, independent of the kernel's max/exp/normalization.
    p = prefix_lse.double().masked_fill(prefix_lse.isinf(), -float("inf"))
    s = suffix_lse.double().masked_fill(suffix_lse.isinf(), -float("inf"))
    lse = torch.logaddexp(p, s)
    pw = (p - lse).exp().nan_to_num(0).T[..., None]
    sw = (s - lse).exp().nan_to_num(0).T[..., None]
    output = torch.where(pw == 0, 0, prefix.double() * pw)
    output += torch.where(sw == 0, 0, suffix.double() * sw)
    return output, lse


def merge_checks(lib):
    fn = lib.merge_attn_states_cuda
    results = []
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            atol, rtol = {
                torch.float32: (2e-6, 2e-5),
                torch.float16: (0.001, 0.001),
                torch.bfloat16: (0.008, 0.008),
            }[dtype]
            for tokens, heads, dim in ((1, 1, 8), (7, 3, 24), (65, 5, 128)):
                for mode in ("finite", "empty-and-extreme"):
                    p = torch.randn(
                        tokens, heads, dim, device="cuda", dtype=dtype
                    )
                    s = torch.randn_like(p)
                    pl = torch.randn(heads, tokens, device="cuda") * 5
                    sl = torch.randn_like(pl) * 5
                    if mode == "empty-and-extreme":
                        pf, sf = pl.view(-1), sl.view(-1)
                        pf[::4], sf[::4] = -float("inf"), float("inf")
                        pf[1::4] = float("inf")
                        sf[2::4] = -float("inf")
                        pf[3::4], sf[3::4] = 1000.0, -1000.0
                        p[pl.isinf().T] = float("nan")
                        s[sl.isinf().T] = float("nan")
                    expected, expected_lse = merge_reference(p, pl, s, sl)
                    for with_lse in (False, True):
                        out = torch.full_like(p, float("nan"))
                        lse = torch.empty_like(pl) if with_lse else None
                        fn(out, lse, p, pl, s, sl)
                        torch.testing.assert_close(
                            out, expected.to(dtype), atol=atol, rtol=rtol
                        )
                        if with_lse:
                            torch.testing.assert_close(
                                lse, expected_lse.float(), atol=2e-6, rtol=2e-6
                            )
                        results.append(
                            {
                                "function": "merge_attn_states_cuda",
                                "shape_thd": [tokens, heads, dim],
                                "dtype": str(dtype),
                                "case": mode,
                                "with_output_lse": with_lse,
                                "max_abs_error": (out.double() - expected)
                                .abs()
                                .max()
                                .item(),
                                "atol": atol,
                                "rtol": rtol,
                                "stream": "non-default",
                                "passed": True,
                            }
                        )
            # Validate composition from independently computed attention states.
            q = torch.randn(7, 3, 24, device="cuda", dtype=dtype) * 0.5
            k, v = (
                torch.randn(11, 3, 24, device="cuda", dtype=dtype) * 0.5
                for _ in range(2)
            )
            scores = (
                torch.einsum("thd,shd->hts", q.double(), k.double()) / 24**0.5
            )
            p = (
                torch.einsum(
                    "hts,shd->thd", scores[..., :4].softmax(-1), v[:4].double()
                )
                .to(dtype)
                .contiguous()
            )
            s = (
                torch.einsum(
                    "hts,shd->thd", scores[..., 4:].softmax(-1), v[4:].double()
                )
                .to(dtype)
                .contiguous()
            )
            pl = scores[..., :4].logsumexp(-1).float().contiguous()
            sl = scores[..., 4:].logsumexp(-1).float().contiguous()
            out, lse = torch.empty_like(p), torch.empty_like(pl)
            fn(out, lse, p, pl, s, sl)
            expected = torch.einsum(
                "hts,shd->thd", scores.softmax(-1), v.double()
            )
            torch.testing.assert_close(
                out, expected.to(dtype), atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                lse, scores.logsumexp(-1).float(), atol=2e-6, rtol=2e-6
            )
            results.append(
                {
                    "function": "merge_attn_states_cuda",
                    "case": "split-KV-composition",
                    "dtype": str(dtype),
                    "passed": True,
                }
            )

        out = torch.empty(7, 3, 24, device="cuda")
        p, s = torch.randn_like(out), torch.randn_like(out)
        pl, sl = (torch.randn(3, 7, device="cuda") for _ in range(2))
        bad = {
            "cpu-state": lambda: fn(out, None, p.cpu(), pl, s, sl),
            "state-dtype": lambda: fn(out, None, p.half(), pl, s, sl),
            "lse-dtype": lambda: fn(out, None, p, pl.half(), s, sl),
            "lse-shape": lambda: fn(out, None, p, pl.T.contiguous(), s, sl),
            "noncontiguous": lambda: fn(
                out, None, p.transpose(0, 1), pl, s, sl
            ),
            "out-alias": lambda: fn(p, None, p, pl, s, sl),
            "lse-alias": lambda: fn(out, pl, p, pl, s, sl),
            "pack-alignment": lambda: fn(
                out[..., :23].contiguous(),
                None,
                p[..., :23].contiguous(),
                pl,
                s[..., :23].contiguous(),
                sl,
            ),
            "unaligned-storage": lambda: fn(
                out,
                None,
                torch.empty(out.numel() + 1, device="cuda")[1:].view_as(out),
                pl,
                s,
                sl,
            ),
            "autograd": lambda: fn(
                out, None, p.clone().requires_grad_(), pl, s, sl
            ),
        }
        for name, call in bad.items():
            try:
                call()
            except RuntimeError:
                results.append(
                    {
                        "function": "merge_attn_states_cuda",
                        "case": name,
                        "passed": True,
                    }
                )
            else:
                raise AssertionError(f"{name}: invalid inputs were accepted")
        empty = torch.empty(0, 3, 24, device="cuda")
        el = torch.empty(3, 0, device="cuda")
        fn(empty, None, empty.clone(), el, empty.clone(), el)
        results.append(
            {
                "function": "merge_attn_states_cuda",
                "case": "zero-tokens",
                "passed": True,
            }
        )
        if torch.cuda.device_count() > 1:
            device = torch.cuda.current_device()
            other = 0 if device != 0 else 1
            expected, _ = merge_reference(p, pl, s, sl)
            torch.cuda.set_device(other)
            try:
                fn(out, None, p, pl, s, sl)
                assert torch.cuda.current_device() == other
                torch.testing.assert_close(
                    out, expected.float(), atol=2e-6, rtol=2e-5
                )
            finally:
                torch.cuda.set_device(device)
            results.append(
                {
                    "function": "merge_attn_states_cuda",
                    "case": "device-guard-restores-caller",
                    "passed": True,
                }
            )
            try:
                fn(out, None, p.to(f"cuda:{other}"), pl, s, sl)
            except RuntimeError:
                results.append(
                    {
                        "function": "merge_attn_states_cuda",
                        "case": "mixed-devices-rejected",
                        "passed": True,
                    }
                )
            else:
                raise AssertionError("mixed devices were accepted")
    stream.synchronize()
    return results
