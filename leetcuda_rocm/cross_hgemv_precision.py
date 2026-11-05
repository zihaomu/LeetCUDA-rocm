"""Compare explicit AMD WMMA precision choices with frozen CUDA outputs."""

import argparse
import json
from datetime import datetime, timezone

import torch

from .cross_validate import ROOT, binding, sha, timed
from .environment import cache_key, verify
from .isa import hip_isa
from .loader import _load, _prepare


def main():
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--cuda", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from pathlib import Path

    fixture = torch.load(args.fixtures, weights_only=True)
    cuda_report = json.loads(Path(args.cuda).read_text())
    assert (
        sha(Path(args.cuda).with_suffix(".pt")) == cuda_report["outputs_sha256"]
    )
    cuda = torch.load(Path(args.cuda).with_suffix(".pt"), weights_only=True)
    cuda_rows = {r["id"]: r for r in cuda_report["checks"]}
    identity = verify()
    source = ROOT / "kernels/rocm/hgemv_matrix_precision.cu"
    names = ("hgemv_wmma_tile16_half_state", "hgemv_wmma_fp32_state")
    lib = _load(
        "hgemv_matrix_precision",
        (_prepare(binding(names)), _prepare(source.read_text())),
        cache_key(identity),
        identity["gpu_architecture"],
        False,
    )
    checks = []
    for c in fixture["cases"]:
        if c["module"] != "hgemv":
            continue
        key = c["id"] + "/hgemv_tensor_core_cute"
        assert cuda_rows[key]["input_sha256"] == c["input_sha256"]
        a, b = (t.cuda() for t in c["inputs"])
        for name in ("hgemv_wmma_tile16_half_state", "hgemv_wmma_fp32_state"):
            fn = getattr(lib, name)
            out = torch.empty_like(a[:, :1]).contiguous()
            fn(a, b, out)
            actual = out.cpu()
            expected = c["reference"].half()
            row = dict(
                id=c["id"],
                function=name,
                fp64_accuracy_passed=torch.allclose(
                    actual, expected, atol=c["atol"], rtol=c["rtol"]
                ),
                atol=c["atol"],
                rtol=c["rtol"],
                max_abs_error_fp64=(actual.double() - c["reference"])
                .abs()
                .max()
                .item(),
                matches_cuda=torch.equal(actual, cuda[key]),
                max_abs_cuda=(actual.double() - cuda[key].double())
                .abs()
                .max()
                .item(),
                matches_half_tile_model=torch.equal(
                    actual, c["models"]["hgemv_tensor_core_cute"]
                ),
            )
            if "20261105" in c["id"]:
                row["performance"] = timed(lambda: fn(a, b, out))
            checks.append(row)
    isa = hip_isa(
        lib.__file__, ROOT / "artifacts/rocm/cross/hgemv_precision_isa"
    )
    assert "v_wmma_f32_16x16x16_f16" in isa["instructions"]
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        environment=identity,
        fixture_sha256=sha(args.fixtures),
        cuda_report_sha256=sha(args.cuda),
        source_sha256={
            str(p.relative_to(ROOT)): sha(p) for p in (source, Path(__file__))
        },
        binary_sha256=sha(lib.__file__),
        checks=checks,
        isa=isa,
        note="Research only. FP32-state changes accumulation precision; it is not a bitwise FP16-accumulator port.",
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    for name in {r["function"] for r in checks}:
        selected = [r for r in checks if r["function"] == name]
        print(
            name,
            "cases",
            len(selected),
            "accuracy",
            sum(r["fp64_accuracy_passed"] for r in selected),
            "CUDA equal",
            sum(r["matches_cuda"] for r in selected),
            flush=True,
        )


if __name__ == "__main__":
    main()
