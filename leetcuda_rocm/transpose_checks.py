"""Bit-preserving layout, tail, stream and contract checks for AMD transposes."""

import torch


def transpose_checks(lib):
    results = []
    device = torch.cuda.current_device()
    names = sorted(n for n in dir(lib) if n.startswith("mat_transpose_cute"))
    for name in names:
        fn = getattr(lib, name)
        for stream in (torch.cuda.default_stream(), torch.cuda.Stream()):
            with torch.cuda.stream(stream):
                # Include signed zero, infinities and distinct NaN payloads.
                bits = torch.tensor(
                    [
                        0,
                        -2147483648,
                        1065353216,
                        2139095040,
                        -8388608,
                        2143289345,
                        -4194302,
                    ],
                    device="cuda",
                    dtype=torch.int32,
                )
                for m, n in (
                    (1, 1),
                    (3, 5),
                    (16, 16),
                    (17, 63),
                    (64, 128),
                    (65, 129),
                    (0, 7),
                    (7, 0),
                ):
                    x = (
                        bits.repeat((m * n + 6) // 7)[: m * n]
                        .view(torch.float32)
                        .view(m, n)
                    )
                    out = torch.full((n, m), 42.0, device="cuda")
                    fn(x, out)
                    assert torch.equal(
                        out.view(torch.int32),
                        x.T.contiguous().view(torch.int32),
                    )
                    results.append(
                        {
                            "function": name,
                            "case": "bitwise-special-values-and-tails",
                            "shape": [m, n],
                            "stream": (
                                "default"
                                if stream == torch.cuda.default_stream()
                                else "non-default"
                            ),
                            "passed": True,
                        }
                    )
            stream.synchronize()
        x = torch.randn(16, 16, device="cuda")
        out = torch.empty_like(x)
        bad = {
            "cpu": lambda: fn(x.cpu(), out),
            "dtype": lambda: fn(x.half(), out),
            "shape": lambda: fn(x, out[:, :15]),
            "noncontiguous": lambda: fn(x.T, out),
            "alias": lambda: fn(x, x),
            "autograd": lambda: fn(x.clone().requires_grad_(), out),
        }
        if "vectorized" in name:
            bad["unaligned-input"] = lambda: fn(
                torch.empty(257, device="cuda")[1:].view_as(x), out
            )
            bad["unaligned-output"] = lambda: fn(
                x, torch.empty(257, device="cuda")[1:].view_as(out)
            )
        if torch.cuda.device_count() > 1:
            other = 0 if device != 0 else 1
            bad["mixed-devices"] = lambda: fn(x, out.to(f"cuda:{other}"))
            torch.cuda.synchronize(device)
            with torch.cuda.device(other):
                fn(x, out)
                assert torch.cuda.current_device() == other
            torch.cuda.synchronize(device)
            assert torch.equal(out, x.T)
            results.append(
                {
                    "function": name,
                    "case": "device-guard-restores-caller",
                    "passed": True,
                }
            )
        for case, call in bad.items():
            try:
                call()
            except RuntimeError:
                results.append(
                    {"function": name, "case": f"reject-{case}", "passed": True}
                )
            else:
                raise AssertionError(f"{name}: accepted invalid {case}")
    return results
