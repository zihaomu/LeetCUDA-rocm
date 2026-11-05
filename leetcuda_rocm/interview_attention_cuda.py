"""Test-only compilation of original interview attention, one kind/D per DSO."""

import hashlib, json, os
from pathlib import Path
from .environment import ROOT
from .cross_validate import CUTLASS, UPSTREAM, sha


def build(kind, d, precision=1, slotfix=False, original=None, cutlass=None):
    from torch.utils.cpp_extension import load

    original = Path(original or ROOT / "original")
    cutlass = Path(cutlass or ROOT / ".cache/cutlass")
    source = (
        ROOT / "leetcuda_rocm/native/interview_attention_cuda_reference.cpp"
    )
    paths = [
        source,
        *[
            original / "kernels/interview" / p
            for p in ("hgemm.cuh", "flash_attn.cuh", "base.cuh", "common.cuh")
        ],
    ]
    hashes = {str(p.resolve()): sha(p) for p in paths}
    flags = [
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-DNOTES_V2_ENABLE_CUTE=1",
        "-DNOTES_V2_ENABLE_TMA_MMA_WS=1",
        f"-DINTERVIEW_KIND={kind}",
        f"-DINTERVIEW_D={d}",
        f"-DINTERVIEW_ACC_F32={precision}",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]
    binding = source.read_text()
    staged_header = None
    if d in (32, 96):
        binding = binding.replace(
            "void launch(Tensor q,Tensor k,Tensor v,Tensor o,Tensor maps,bool causal,float scale){",
            "void launch(Tensor q,Tensor k,Tensor v,Tensor o,Tensor maps,bool causal,float scale){\n constexpr int D=INTERVIEW_D+0*SK;",
        )
        binding = binding.replace(
            " else if constexpr(Kind==6){",
            " else if constexpr(Kind==6){\n#if INTERVIEW_KIND == 6",
        ).replace(
            " }else{\n  TORCH_CHECK(sv==1",
            "\n#endif\n }else{\n  TORCH_CHECK(sv==1",
        )
        hashes["generated_binding_sha256"] = hashlib.sha256(
            binding.encode()
        ).hexdigest()
    if slotfix and kind == 0:
        staged_header = (
            original / "kernels/interview/flash_attn.cuh"
        ).read_text()
        end = staged_header.index("// ★ Shared Memory")
        body = staged_header[:end]
        assert body.count("(tile_K_seqlen + 1)") == 4
        body = body.replace(
            "(tile_K_seqlen + 1)", "(tile_K_seqlen + kStagesK - 1)"
        )
        assert body.count("stage < (kStagesK - 1)") == 1
        body = body.replace(
            "stage < (kStagesK - 1)", "stage < (kStagesK - 1) && stage < Tc"
        )
        assert body.count("CP_ASYNC_WAIT_GROUP(kStagesK - 2);") == 1
        body = body.replace(
            "CP_ASYNC_WAIT_GROUP(kStagesK - 2);", "CP_ASYNC_WAIT_GROUP(0);"
        )
        staged_header = body + staged_header[end:]
        binding = binding.replace(
            '#include "flash_attn.cuh"', '#include "flash_attn_slotfix.cuh"'
        )
        hashes["explicit_slotfix_header_sha256"] = hashlib.sha256(
            staged_header.encode()
        ).hexdigest()
        hashes["generated_binding_sha256"] = hashlib.sha256(
            binding.encode()
        ).hexdigest()
    if slotfix and kind == 7:
        staged_header = (
            original / "kernels/interview/flash_attn.cuh"
        ).read_text()
        start = staged_header.index("CUTE_DEVICE void online_safe_softmax_fa4(")
        end = staged_header.index("}  // namespace fa_cute", start)
        body = staged_header[start:end]
        needle = "    const float next_max = fmaxf(row_max[row], tile_max);"
        assert body.count(needle) == 1
        # A completely masked tile contributes zero probability, even before
        # a finite running maximum exists. Avoid both -inf - -inf and NaN*0.
        replacement = (
            """    if (tile_max == -INFINITY && row_max[row] == -INFINITY) {
      row_scale[row] = 1.0f;
#pragma unroll
      for (int col = 0; col < cute::size<1>(scores); ++col)
        scores(row, col) = 0.0f;
      continue;
    }
"""
            + needle
        )
        staged_header = (
            staged_header[:start]
            + body.replace(needle, replacement)
            + staged_header[end:]
        )
        binding = binding.replace(
            '#include "flash_attn.cuh"', '#include "flash_attn_slotfix.cuh"'
        )
        hashes["explicit_empty_row_fix_header_sha256"] = hashlib.sha256(
            staged_header.encode()
        ).hexdigest()
        hashes["generated_binding_sha256"] = hashlib.sha256(
            binding.encode()
        ).hexdigest()
    digest = hashlib.sha256(
        json.dumps([hashes, flags, CUTLASS], sort_keys=True).encode()
    ).hexdigest()[:16]
    folder = (
        ROOT
        / ".cache/cuda_comparison"
        / f"interview_attention_{kind}_{d}_{digest}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    staged = folder / "reference.cu"
    if not staged.exists() or staged.read_text() != binding:
        staged.write_text(binding)
    if staged_header is not None:
        hp = folder / "flash_attn_slotfix.cuh"
        if not hp.exists() or hp.read_text() != staged_header:
            hp.write_text(staged_header)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    os.environ.setdefault("MAX_JOBS", "2")
    lib = load(
        name=f"leetcuda_interview_attention_{digest}",
        sources=[str(staged)],
        build_directory=str(folder),
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=flags,
        extra_include_paths=[
            str(original / "kernels/interview"),
            str(cutlass / "include"),
        ],
        extra_ldflags=["-lcuda"],
    )
    return lib, dict(
        kind=kind,
        d=d,
        precision=precision,
        source_kind=(
            ("explicit_slotfix" if kind == 0 else "explicit_empty_row_fix")
            if slotfix and kind in (0, 7)
            else "upstream_original"
        ),
        source_sha256=hashes,
        flags=flags,
        cutlass_commit=CUTLASS,
        upstream_commit=UPSTREAM,
        binary=lib.__file__,
        binary_sha256=sha(lib.__file__),
    )
