"""Probe locked hipBLAS compute modes before mapping original cuBLAS APIs."""

import ctypes as ct
import hashlib
import json
from datetime import datetime, timezone

import torch

from .environment import ROOT, verify


def main():
    identity = verify()
    torch.manual_seed(20261103)
    library = ct.CDLL("/opt/rocm/lib/libhipblas.so")
    ptr, integer = ct.c_void_p, ct.c_int
    library.hipblasCreate.argtypes = [ct.POINTER(ptr)]
    library.hipblasDestroy.argtypes = [ptr]
    library.hipblasSetStream.argtypes = [ptr, ptr]
    library.hipblasSetMathMode.argtypes = [ptr, integer]
    library.hipblasGemmEx.argtypes = [
        ptr,
        integer,
        integer,
        integer,
        integer,
        integer,
        ptr,
        ptr,
        integer,
        integer,
        ptr,
        integer,
        integer,
        ptr,
        ptr,
        integer,
        integer,
        integer,
        integer,
    ]
    handle = ptr()
    assert library.hipblasCreate(ct.byref(handle)) == 0
    checks = []
    try:
        assert (
            library.hipblasSetStream(
                handle, torch.cuda.current_stream().cuda_stream
            )
            == 0
        )
        tf32_status = library.hipblasSetMathMode(handle, 3)
        assert library.hipblasSetMathMode(handle, 0) == 0
        for dtype, datatype, compute in (
            (torch.float32, 0, 2),
            (torch.float16, 2, 0),
            (torch.float16, 2, 1),
            (torch.float32, 0, 6),
        ):
            alpha = (
                ct.c_uint16(0x3C00) if dtype == torch.float16 else ct.c_float(1)
            )
            beta = ct.c_uint16(0) if dtype == torch.float16 else ct.c_float(0)
            for m, n, k in ((17, 33, 19), (128, 256, 128), (256, 256, 256)):
                for transposed_b in (False, True):
                    a = torch.randn(m, k, device="cuda", dtype=dtype) * 0.25
                    b = torch.randn(k, n, device="cuda", dtype=dtype) * 0.25
                    packed_b = b.T.contiguous() if transposed_b else b
                    out = torch.full(
                        (m, n), float("nan"), device="cuda", dtype=dtype
                    )
                    status = library.hipblasGemmEx(
                        handle,
                        112 if transposed_b else 111,
                        111,
                        n,
                        m,
                        k,
                        ct.byref(alpha),
                        packed_b.data_ptr(),
                        datatype,
                        k if transposed_b else n,
                        a.data_ptr(),
                        datatype,
                        k,
                        ct.byref(beta),
                        out.data_ptr(),
                        datatype,
                        n,
                        compute,
                        160,
                    )
                    record = {
                        "dtype": str(dtype),
                        "compute_enum": compute,
                        "shape_mnk": [m, n, k],
                        "transposed_b_storage": transposed_b,
                        "hipblas_status": status,
                    }
                    if status == 0:
                        expected = a.double() @ b.double()
                        tol = 0.01 if dtype == torch.float16 else 2e-4
                        record.update(
                            max_abs_error=(out.double() - expected)
                            .abs()
                            .max()
                            .item(),
                            atol=tol,
                            rtol=tol,
                            accuracy_passed=torch.allclose(
                                out, expected.to(dtype), atol=tol, rtol=tol
                            ),
                        )
                    checks.append(record)
                    print(json.dumps(record), flush=True)
        torch.cuda.synchronize()
    finally:
        assert library.hipblasDestroy(handle) == 0
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": identity,
        "seed": 20261103,
        "tf32_math_mode_status": tf32_status,
        "checks": checks,
        "compute_enum_names": {
            0: "16F",
            1: "16F_PEDANTIC",
            2: "32F",
            6: "32F_FAST_TF32",
        },
        "qualified": False,
        "source_sha256": hashlib.sha256(
            open(__file__, "rb").read()
        ).hexdigest(),
    }
    output = ROOT / "artifacts/rocm/blas/research.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
