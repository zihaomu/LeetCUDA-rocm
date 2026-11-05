"""Establish a CUDA oracle for every still-pending original interface.

Compile independent source groups in subprocesses, then execute groups serially.
Failed compilation, launch, and numerical gates remain explicit in the report.
This does not imply an AMD implementation exists for any pending interface.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from .cross_validate import CUTLASS, ROOT, UPSTREAM, sha, tensor_sha, timed


def plan(args):
    inventory = json.loads(args.inventory.read_text())
    pending = [e for e in inventory["entries"] if e["status"] == "pending"]
    groups = {}
    for entry in pending:
        module_root = (
            args.source_root / "kernels" / entry["source"].split("/")[1]
        )
        # The source tree includes unbound experimental files that reuse export
        # names (e.g. HGEMM x2 versus x4). Honor the original build source list.
        build_script = {
            "hgemm": module_root / "tools/utils.py",
            "flash_attn": module_root / "flash_attn_mma.py",
        }.get(entry["module"])
        allowed = None
        if build_script:
            allowed = {
                (module_root / p).resolve()
                for p in re.findall(
                    r'build_sources\.append\(\s*"([^"]+)"\s*\)',
                    build_script.read_text(),
                )
            }
        found = []
        for path in module_root.rglob("*.cu"):
            if allowed is not None and path.resolve() not in allowed:
                continue
            text = path.read_text(encoding="utf-8-sig")
            match = re.search(
                r"\bvoid\s+" + entry["function"] + r"\s*\(([^)]*)\)\s*\{", text
            )
            if match:
                found.append((path, match))
        assert len(found) == 1, (entry["function"], [str(p) for p, _ in found])
        path, match = found[0]
        relative = str(path.relative_to(args.source_root))
        record = groups.setdefault(
            relative,
            {"source": relative, "module": entry["module"], "functions": []},
        )
        record["functions"].append(
            dict(
                name=entry["function"],
                declaration=f"void {entry['function']}({match[1]});",
                argc=len(match[1].split(",")),
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        groups.values(),
        key=lambda g: (
            {"sgemm": 0, "ws_hgemm": 1, "hgemm": 2, "flash_attn": 3}[
                g["module"]
            ],
            g["source"],
        ),
    )
    args.output.write_text(
        json.dumps(
            dict(
                upstream_commit=UPSTREAM,
                inventory_sha256=sha(args.inventory),
                groups=ordered,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        f"{len(pending)} pending interfaces in {len(groups)} source groups",
        flush=True,
    )


def fixtures(path):
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(20261105)

    def rand(shape, dtype):
        return (torch.randn(shape, generator=generator) * 0.25).to(dtype)

    cases = {}
    for module in ("hgemm", "sgemm", "ws_hgemm"):
        dtype = torch.float32 if module == "sgemm" else torch.float16
        cases[module] = []
        for index, (m, n, k) in enumerate(
            ((256, 256, 256), (256, 512, 512), (512, 512, 128))
        ):
            a, b = rand((m, k), dtype), rand((k, n), dtype)
            cases[module].append(
                dict(
                    id=f"{module}_{m}_{n}_{k}",
                    inputs=(a, b),
                    reference=a.double() @ b.double(),
                    stage=index + 2,
                    swizzle=index == 1,
                )
            )
    cases["flash_attn"] = []
    for index, (seq, dim) in enumerate(((128, 32), (256, 64), (256, 128))):
        q, k, v = (rand((1, 2, seq, dim), torch.float16) for _ in range(3))
        reference = (
            (q.double() @ k.double().transpose(-1, -2)) / math.sqrt(dim)
        ).softmax(-1) @ v.double()
        cases["flash_attn"].append(
            dict(
                id=f"attention_{seq}_{dim}",
                inputs=(q, k, v),
                reference=reference,
                stage=2,
            )
        )
    for entries in cases.values():
        for c in entries:
            c["input_sha256"] = [tensor_sha(t) for t in c["inputs"]]
    torch.save(cases, path)


def build(group, args):
    from torch.utils.cpp_extension import load

    source = args.source_root / group["source"]
    module_root = args.source_root / "kernels" / group["source"].split("/")[1]
    headers = sorted(module_root.rglob("*.h"))
    sources = [str(source)]
    flags = [
        "-O3",
        "-std=c++20",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-DNO_CUBLAS_HGEMM_BIN",
        "-DNO_MMA_HGEMM_BIN",
        "-DNO_WMMA_HGEMM_BIN",
        "-DNO_CUTE_HGEMM_BIN",
        "-DBUILD_FLASH_ATTN_MMA_OTHERS",
    ]
    source_hashes = {
        str(p.relative_to(args.source_root)): sha(p) for p in (source, *headers)
    }
    current_stream = getattr(args, "cuda_current_stream", False)
    fingerprint_data = [group, source_hashes, flags, CUTLASS]
    if current_stream:
        fingerprint_data.append("current-stream-v1")
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_data, sort_keys=True).encode()
    ).hexdigest()[:16]
    directory = ROOT / ".cache/cuda_pending" / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    adapted_sha = None
    if current_stream:
        # Opt-in for graph timing. Preserve the original kernel arithmetic,
        # but use the caller's CUDA stream instead of the legacy default.
        def stream_launch(match):
            fields = match[1].split(",")
            if len(fields) == 2:
                fields.append("0")
            assert len(fields) == 3, match[0]
            return (
                "<<<"
                + ",".join(fields)
                + ", at::cuda::getCurrentCUDAStream()>>>"
            )

        adapted = "#include <ATen/cuda/CUDAContext.h>\n" + re.sub(
            r"<<<([^<>]+)>>>",
            stream_launch,
            source.read_text(encoding="utf-8-sig"),
        )
        staged = directory / "current_stream.cu"
        if not staged.exists() or staged.read_text() != adapted:
            staged.write_text(adapted)
        sources[0] = str(staged)
        adapted_sha = sha(staged)
    if "PYBIND11_MODULE(" not in source.read_text(encoding="utf-8-sig"):
        binding = "#include <torch/extension.h>\n" + "\n".join(
            f["declaration"] for f in group["functions"]
        )
        binding += (
            "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n"
            + "\n".join(
                f'm.def("{f["name"]}",&{f["name"]});'
                for f in group["functions"]
            )
            + "\n}\n"
        )
        path = directory / "binding.cpp"
        if not path.exists() or path.read_text() != binding:
            path.write_text(binding)
        sources.append(str(path))
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
    os.environ["MAX_JOBS"] = "2"
    include = [
        module_root,
        module_root / "utils",
        args.cutlass / "include",
        args.cutlass / "tools/util/include",
    ]
    lib = load(
        name="leetcuda_pending_" + fingerprint,
        sources=sources,
        build_directory=str(directory),
        extra_cuda_cflags=flags,
        extra_cflags=["-O3", "-std=c++20"],
        extra_include_paths=[str(p) for p in include],
        extra_ldflags=["-lcublas"],
    )
    return lib, dict(
        source_sha256=source_hashes,
        current_stream_adaptation=current_stream,
        adapted_source_sha256=adapted_sha,
        flags=flags,
        cutlass_commit=CUTLASS,
        binary_sha256=sha(lib.__file__),
        binary=lib.__file__,
    )


def worker(args):
    group = json.loads(args.manifest.read_text())["groups"][args.group]
    report = dict(
        group=group,
        phase="compile",
        checks=[],
        error=None,
        passed=False,
        source_sha256=sha(args.source_root / group["source"]),
    )
    output = args.output / f"group_{args.group}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    try:
        lib, metadata = build(group, args)
        report["build"] = metadata
        report["phase"] = "compiled"
        output.write_text(json.dumps(report, indent=2) + "\n")
        if args.build_only:
            return 0
        torch.set_grad_enabled(False)
        fixture = torch.load(args.fixtures, weights_only=True)
        report["phase"] = "execute"
        for function in group["functions"]:
            name = function["name"]
            fn = getattr(lib, name)
            for c in fixture[group["module"]]:
                assert c["input_sha256"] == [tensor_sha(t) for t in c["inputs"]]
                inputs = [t.cuda() for t in c["inputs"]]
                layout = "row_major"
                if name in {
                    "flash_attn_mma_stages_split_q_shared_kv_swizzle_qkv",
                    "flash_attn_mma_stages_split_q_shared_qkv_swizzle_qkv",
                    "flash_attn_mma_stages_split_q_tiling_qk_swizzle_qkv",
                }:
                    # Match flash_attn_mma.py: only these three exports take tv.
                    inputs[2] = inputs[2].transpose(-2, -1).contiguous()
                    layout = "Q,K:[B,H,N,D]; V:[B,H,D,N]"
                if group["module"] == "ws_hgemm" or (
                    group["module"] == "hgemm"
                    and ("_tn_" in name or name.endswith("_tn"))
                ):
                    inputs[1] = inputs[1].T.contiguous().view_as(inputs[1])
                    layout = "B_storage:[N,K]; B_shape:[K,N]"
                out = torch.full(
                    c["reference"].shape,
                    float("nan"),
                    device="cuda",
                    dtype=inputs[0].dtype,
                )
                call_args = [*inputs, out]
                if group["module"] == "flash_attn" and function["argc"] == 5:
                    call_args.append(c["stage"])
                elif function["argc"] == 6:
                    call_args.extend((c["stage"], c["swizzle"], 256))
                assert len(call_args) == function["argc"], (
                    name,
                    len(call_args),
                    function["argc"],
                )
                torch.cuda.synchronize()
                fn(*call_args)
                torch.cuda.synchronize()
                actual = out.cpu()
                expected = c["reference"].to(actual.dtype)
                tol = {
                    "hgemm": 0.01,
                    "ws_hgemm": 0.01,
                    "sgemm": 0.005,
                    "flash_attn": 0.003,
                }[group["module"]]
                error = (actual.double() - c["reference"]).abs().max().item()
                row = dict(
                    id=c["id"] + "/" + name,
                    function=name,
                    input_sha256=c["input_sha256"],
                    kernel_input_layout=layout,
                    kernel_input_sha256=[tensor_sha(t.cpu()) for t in inputs],
                    passed=torch.allclose(actual, expected, atol=tol, rtol=tol),
                    atol=tol,
                    rtol=tol,
                    max_abs_error_fp64=error if math.isfinite(error) else None,
                    finite=bool(actual.isfinite().all()),
                    stage=c["stage"],
                    source=group["source"],
                )
                row["performance"] = timed(lambda: fn(*call_args))
                report["checks"].append(row)
                outputs[row["id"]] = actual
                output.write_text(json.dumps(report, indent=2) + "\n")
        report["phase"] = "finished"
        report["passed"] = all(r["passed"] for r in report["checks"])
    except Exception as exc:
        report["error"] = str(exc)[-24000:]
    torch.save(outputs, output.with_suffix(".pt"))
    report["outputs_sha256"] = sha(output.with_suffix(".pt"))
    output.write_text(json.dumps(report, indent=2) + "\n")
    return int(not report["passed"])


def suite(args):
    assert torch.version.hip is None and torch.cuda.get_device_capability() == (
        12,
        0,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    fixture_path = args.output / "fixtures.pt"
    if not fixture_path.exists():
        fixtures(fixture_path)
    groups = json.loads(args.manifest.read_text())["groups"]

    def execute(index, compile_only):
        command = [
            sys.executable,
            "-m",
            "leetcuda_rocm.cuda_pending_check",
            "worker",
            "--manifest",
            str(args.manifest),
            "--group",
            str(index),
            "--source-root",
            str(args.source_root),
            "--cutlass",
            str(args.cutlass),
            "--fixtures",
            str(fixture_path),
            "--output",
            str(args.output),
        ]
        if compile_only:
            command.append("--build-only")
        logfile = (
            args.output
            / f"group_{index}_{'build' if compile_only else 'run'}.log"
        )
        with logfile.open("w") as log:
            try:
                result = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=900
                )
                if compile_only:
                    print(
                        f"compiled group {index + 1}/{len(groups)}: {groups[index]['source']} result={result.returncode}",
                        flush=True,
                    )
                return index, result.returncode
            except subprocess.TimeoutExpired:
                return index, "timeout"

    # Compiler processes only: no timed GPU work overlaps across groups.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.compile_workers
    ) as pool:
        compiled = dict(
            pool.map(lambda i: execute(i, True), range(len(groups)))
        )
    executed = {}
    for index in range(len(groups)):
        if compiled[index] == 0:
            _, code = execute(index, False)
            executed[index] = code
            print(
                f"group {index + 1}/{len(groups)}: {groups[index]['source']} result={code}",
                flush=True,
            )
        else:
            print(
                f"group {index + 1}/{len(groups)}: compile failed ({compiled[index]})",
                flush=True,
            )
    reports = []
    for index, group in enumerate(groups):
        path = args.output / f"group_{index}.json"
        if path.exists():
            group_report = json.loads(path.read_text())
            if executed.get(index) != 0:
                group_report["passed"] = False
                group_report["worker_returncode"] = executed.get(index)
            reports.append(group_report)
        else:
            reports.append(
                dict(
                    group=group,
                    error="worker terminated without a report",
                    checks=[],
                    passed=False,
                )
            )
    report = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        upstream_commit=UPSTREAM,
        fixture_sha256=sha(fixture_path),
        manifest_sha256=sha(args.manifest),
        harness_sha256=sha(__file__),
        environment=dict(
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            image=os.environ["LEETCUDA_CUDA_IMAGE"],
            gpu=torch.cuda.get_device_name(),
            gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        ),
        scope="CUDA oracle only for the 56 AMD-pending interfaces; staged numerical/performance cases, not AMD qualification",
        groups=reports,
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    planner = sub.add_parser("plan")
    planner.add_argument("--inventory", type=Path, required=True)
    planner.add_argument("--source-root", type=Path, default=ROOT)
    planner.add_argument("--output", type=Path, required=True)
    for name in ("suite", "worker"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--manifest", type=Path, required=True)
        cmd.add_argument("--source-root", type=Path, default=ROOT / "original")
        cmd.add_argument(
            "--cutlass", type=Path, default=ROOT / ".cache/cutlass"
        )
        cmd.add_argument("--output", type=Path, required=True)
        if name == "suite":
            cmd.add_argument("--compile-workers", type=int, default=4)
        else:
            cmd.add_argument("--group", type=int, required=True)
            cmd.add_argument("--fixtures", type=Path, required=True)
            cmd.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    return {"plan": plan, "suite": suite, "worker": worker}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
