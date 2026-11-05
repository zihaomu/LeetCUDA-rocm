"""Immutable CUDA swizzle kernels and an explicitly widened FP32 reference."""

import hashlib, json, os, re
from pathlib import Path
from .environment import ROOT
from .cross_validate import sha, UPSTREAM
from .swizzle_examples import NAMES

F32_MACRO = r"""
#define HMMA16816F32(D0,D1,D2,D3,A0,A1,A2,A3,B0,B1,C0,C1,C2,C3) \
asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};" : "=r"(D0),"=r"(D1),"=r"(D2),"=r"(D3) : "r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),"r"(C0),"r"(C1),"r"(C2),"r"(C3))
"""


def kernel_bounds(text, name):
    masked = re.sub(
        r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"',
        lambda m: " " * len(m[0]),
        text,
        flags=re.S,
    )
    pos = masked.index(name + "(")
    start = masked.rfind("template <", 0, pos)
    body = masked.index("{", pos)
    level = 1
    end = body + 1
    while level:
        level += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    return start, body, end


def widened(block):
    start = block.index("uint32_t RC[")
    # The first accumulator read in the epilogue lies after the complete K loop.
    loop = block.index("for (int k = 0;", start)
    brace = block.index("{", loop)
    level = 1
    end = brace + 1
    while level:
        level += (block[end] == "{") - (block[end] == "}")
        end += 1
    compute = block[start:end]
    large = "RC[WARP_TILE_M][WARP_TILE_N][2]" in compute
    compute = re.sub(r"\bRC\b", "RF32", compute)
    if large:
        compute = compute.replace(
            "RF32[WARP_TILE_M][WARP_TILE_N][2];",
            "RF32[WARP_TILE_M][WARP_TILE_N][4] = {};",
        )
    else:
        compute = compute.replace(
            "RF32[2] = {0, 0};", "RF32[4] = {0, 0, 0, 0};"
        )

    def mma(m):
        args = [x.strip() for x in m[1].split(",")]
        assert len(args) == 10
        extra = lambda x, i: x.rsplit("[", 1)[0] + f"[{i}]"
        return (
            "HMMA16816F32("
            + ",".join(
                args[:2]
                + [extra(args[0], 2), extra(args[0], 3)]
                + args[2:8]
                + args[8:]
                + [extra(args[8], 2), extra(args[8], 3)]
            )
            + ");"
        )

    compute, count = re.subn(r"HMMA16816\((.*?)\);", mma, compute, flags=re.S)
    assert count == 1
    shape = "[WARP_TILE_M][WARP_TILE_N][2]" if large else "[2]"
    prefix = (
        "for(int i=0;i<WARP_TILE_M;i++)for(int j=0;j<WARP_TILE_N;j++)"
        if large
        else ""
    )
    index = "[i][j]" if large else ""
    pack = "\nuint32_t RC" + shape + ";" + prefix + "{\n"
    for part in range(2):
        pack += f"RC{index}[{part}]=uint32_t(__half_as_ushort(__float2half_rn(__uint_as_float(RF32{index}[{part*2}])))) | (uint32_t(__half_as_ushort(__float2half_rn(__uint_as_float(RF32{index}[{part*2+1}]))))<<16);\n"
    pack += "}\n"
    return block[:start] + compute + pack + block[end:]


def build(precision="original"):
    from torch.utils.cpp_extension import load

    paths = [
        ROOT / "original/kernels/swizzle" / n
        for n in ("hgemm_mma_swizzle.cu", "mma_simple_swizzle.cu")
    ]
    main, simple = [p.read_text() for p in paths]
    first = kernel_bounds(main, NAMES[0])[0]
    source = main[:first] + F32_MACRO
    helper = re.search(
        r"__device__ __host__ __forceinline__ int swizzle_A_j\(.*?\n}",
        main,
        flags=re.S,
    )[0]
    source += helper + "\n"
    for i, name in enumerate(NAMES):
        text = main if i < 4 else simple
        start, _, end = kernel_bounds(text, name)
        block = text[start:end]
        block = re.sub(r"//[^\n]*|/\*.*?\*/", "", block, flags=re.S)
        if precision == "fp32":
            block = widened(block)
        if i == 4:
            block = block.replace("swizzle_j(", "swizzle_A_j(")
            source += "\n#define printf(...) ((void)0)\n"
        source += block + "\n"
        if i == 4:
            source += "\n#undef printf\n"
    source += """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
using torch::Tensor;
template<int AP,int BP>void tiled(Tensor a,Tensor b,Tensor o,int kind){
 int m=a.size(0),k=a.size(1),n=b.size(1);dim3 grid(n/128,m/128);auto stream=at::cuda::getCurrentCUDAStream();
 if(kind==1){hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel<16,8,16,2,4,4,4,AP,BP><<<grid,256,0,stream>>>((half*)a.data_ptr(),(half*)b.data_ptr(),(half*)o.data_ptr(),m,n,k);}
 else{hgemm_mma_m16n8k16_mma2x4_warp4x4_smem_swizzle_kernel<16,8,16,2,4,4,4,AP,BP><<<grid,256,0,stream>>>((half*)a.data_ptr(),(half*)b.data_ptr(),(half*)o.data_ptr(),m,n,k);}
}
void forward(Tensor a,Tensor b,Tensor o,int kind,int ap,int bp){
 c10::cuda::CUDAGuard guard(a.device());int m=a.size(0),k=a.size(1),n=b.size(1);bool large=kind==1||kind==3;TORCH_CHECK(m%(large?128:16)==0 && n%(large?128:8)==0 && k%16==0,"original CUDA full tiles only");auto stream=at::cuda::getCurrentCUDAStream();dim3 grid(n/8,m/16);
 if(kind==0){hgemm_mma_m16n8k16_naive_kernel<16,8,16><<<grid,32,0,stream>>>((half*)a.data_ptr(),(half*)b.data_ptr(),(half*)o.data_ptr(),m,n,k);}
 else if(kind==2){hgemm_mma_m16n8k16_naive_smem_swizzle_kernel<16,8,16><<<grid,32,0,stream>>>((half*)a.data_ptr(),(half*)b.data_ptr(),(half*)o.data_ptr(),m,n,k);}
 else if(kind==4){mma_simple_swizzle_kernel<16,8,16><<<grid,32,0,stream>>>((half*)a.data_ptr(),(half*)b.data_ptr(),(half*)o.data_ptr(),m,n,k);}
 else if(ap==8){if(bp==8)tiled<8,8>(a,b,o,kind);else tiled<8,0>(a,b,o,kind);}
 else{if(bp==8)tiled<0,8>(a,b,o,kind);else tiled<0,0>(a,b,o,kind);}C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
"""
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    folder = ROOT / ".cache/cuda_comparison" / ("swizzle_examples_" + digest)
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / "source.cu"
    if not p.exists() or p.read_text() != source:
        p.write_text(source)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    os.environ.setdefault("MAX_JOBS", "2")
    lib = load(
        name="leetcuda_swizzle_examples_" + digest,
        sources=[str(p)],
        build_directory=str(folder),
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
    )
    return lib, dict(
        precision=precision,
        original_sha256={str(p): sha(p) for p in paths},
        reference_source=str(p),
        reference_sha256=sha(p),
        binary=lib.__file__,
        binary_sha256=sha(lib.__file__),
        upstream_commit=UPSTREAM,
        adaptation="current-stream launcher, printing disabled; FP32 arithmetic only when explicitly selected",
    )
