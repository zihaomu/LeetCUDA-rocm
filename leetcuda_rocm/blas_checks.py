"""Independent numerical, layout and lifetime checks for original BLAS APIs."""

import torch


def blas_checks(module, lib):
    names = sorted(
        n for n in dir(lib) if n.startswith(("sgemm_cublas", "hgemm_cublas"))
    )
    results = []
    dtype = torch.float16 if module == "hgemm" else torch.float32
    tol = 0.01 if dtype == torch.float16 else 2e-4
    for name in names:
        fn = getattr(lib, name)
        for seed in (20261103, 20261104, 20261105):
            torch.manual_seed(seed)
            for stream in (torch.cuda.default_stream(), torch.cuda.Stream()):
                with torch.cuda.stream(stream):
                    for m, n, k in (
                        (1, 1, 1),
                        (17, 33, 19),
                        (128, 256, 128),
                        (256, 128, 256),
                        (17, 33, 1025),
                        (0, 5, 8),
                        (3, 0, 7),
                        (3, 4, 0),
                    ):
                        a = torch.randn(m, k, device="cuda", dtype=dtype) * 0.25
                        b = torch.randn(k, n, device="cuda", dtype=dtype) * 0.25
                        packed = (
                            b.T.contiguous().view(k, n)
                            if name.endswith("_tn")
                            else b
                        )
                        out = torch.full(
                            (m, n), float("nan"), device="cuda", dtype=dtype
                        )
                        fn(a, packed, out)
                        expected = a.double() @ b.double()
                        torch.testing.assert_close(
                            out, expected.to(dtype), atol=tol, rtol=tol
                        )
                        results.append(
                            {
                                "function": name,
                                "shape_mnk": [m, n, k],
                                "seed": seed,
                                "reference": "FP64 matrix product",
                                "b_storage": (
                                    "packed [N,K] in [K,N] tensor"
                                    if name.endswith("_tn")
                                    else "row-major [K,N]"
                                ),
                                "stream": (
                                    "default"
                                    if stream == torch.cuda.default_stream()
                                    else "non-default"
                                ),
                                "max_abs_error": (
                                    (out.double() - expected).abs().max().item()
                                    if out.numel()
                                    else 0
                                ),
                                "atol": tol,
                                "rtol": tol,
                                "passed": True,
                            }
                        )
                stream.synchronize()
        a, b, out = (
            torch.randn(16, 16, device="cuda", dtype=dtype) for _ in range(3)
        )
        bad = {
            "cpu": lambda: fn(a.cpu(), b, out),
            "dtype": lambda: fn(a.double(), b, out),
            "shape": lambda: fn(a, b, out[:15]),
            "strides": lambda: fn(a.T, b, out),
            "alias-a": lambda: fn(a, b, a),
            "alias-b": lambda: fn(a, b, b),
            "grad-input": lambda: fn(a.clone().requires_grad_(), b, out),
            "grad-output": lambda: fn(a, b, out.clone().requires_grad_()),
        }
        device = torch.cuda.current_device()
        if torch.cuda.device_count() > 1:
            other = 0 if device != 0 else 1
            bad["mixed-devices"] = lambda: fn(a, b.to(f"cuda:{other}"), out)
            torch.cuda.synchronize(device)
            with torch.cuda.device(other):
                fn(a, b, out)
                assert torch.cuda.current_device() == other
            torch.cuda.synchronize(device)
            expected = (
                a.double() @ (b.T if name.endswith("_tn") else b).double()
            )
            torch.testing.assert_close(
                out, expected.to(dtype), atol=tol, rtol=tol
            )
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
        if module == "hgemm":
            for _ in range(2):
                lib.destroy_cublas_handle()
                lib.init_cublas_handle()
                fn(a, b, out)
                expected = (
                    a.double() @ (b.T if name.endswith("_tn") else b).double()
                )
                torch.testing.assert_close(
                    out, expected.to(dtype), atol=tol, rtol=tol
                )
            results.append(
                {
                    "function": name,
                    "case": "destroy-reinitialize-handle",
                    "passed": True,
                }
            )
    return results
