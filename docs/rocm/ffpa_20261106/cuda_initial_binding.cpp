// Test-only current-stream launcher for unmodified original CUDA kernels.
#include "hgemm.cuh"
#include "ffpa_attn.cuh"
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
using torch::Tensor;
using namespace cute;
using E=cutlass::half_t;
E *ptr(Tensor x){return reinterpret_cast<E *>(x.data_ptr());}

template<int D,int SQ,int SV,int Kind>
void launch(Tensor q,Tensor k,Tensor v,Tensor o,Tensor lse) {
  int nq=q.size(2),nk=k.size(2),bh=q.size(0)*q.size(1);
  auto stream=at::cuda::getCurrentCUDAStream();
  if constexpr(Kind<2) {
    TORCH_CHECK(nq==nk && nq%64==0,"original async/WS requires equal aligned N");
    using T=fa_cute::FFPAAttnSplitDCuTeTraits<D>;
    using LQ=typename T::SmemLayoutQ;using LK=typename T::SmemLayoutKV;
    int bytes=(SQ*(cosize(LQ{})+cosize(LK{}))+SV*cosize(LK{}))*sizeof(E);
    if constexpr(Kind==0) {
      auto f=ffpa_split_d_cute<D,SQ,SV>;
      C10_CUDA_CHECK(cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
      f<<<dim3(nq/64,bh),128,bytes,stream>>>(ptr(q),ptr(k),ptr(v),ptr(o),bh*nq,nq);
    } else {
      auto tensor=[&](Tensor x){return make_tensor(make_gmem_ptr(ptr(x)),make_shape(bh*nq,Int<D>{}),make_stride(Int<D>{},_1{}));};
      auto tq=make_tma_copy(SM90_TMA_LOAD{},tensor(q),LQ{},Shape<_64,_64>{},_1{});
      auto tk=make_tma_copy(SM90_TMA_LOAD{},tensor(k),LK{},Shape<_64,_64>{},_1{});
      auto tv=make_tma_copy(SM90_TMA_LOAD{},tensor(v),LK{},Shape<_64,_64>{},_1{});
      auto f=ffpa_attn_tma_mma_ws_split_d_cute<D,decltype(tq),decltype(tk),decltype(tv),SQ,SV>;
      C10_CUDA_CHECK(cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
      f<<<dim3(nq/64,bh),256,bytes,stream>>>(tq,tk,tv,ptr(o),bh*nq,nq);
    }
  } else {
    TORCH_CHECK(nq%128==0,"original non-WS requires NQ%128=0");
    using T=fa_cute::FFPAAttnNonWSCuTeSplitDTraits<D,128,128,32,64,SQ,SV>;
    auto tensor=[&](Tensor x){return make_tensor(make_gmem_ptr(ptr(x)),make_shape(int(x.size(0)*x.size(1)*x.size(2)),Int<D>{}),make_stride(Int<D>{},_1{}));};
    auto tq=make_tma_copy(SM90_TMA_LOAD{},tensor(q),typename T::SmemLayoutQ{},Shape<_128,_32>{},_1{});
    auto tk=make_tma_copy(SM90_TMA_LOAD{},tensor(k),typename T::SmemLayoutK{},Shape<_128,_32>{},_1{});
    auto tv=make_tma_copy(SM90_TMA_LOAD{},tensor(v),typename T::SmemLayoutV{},Shape<_128,_64>{},_1{});
    auto to=make_tma_copy(SM90_TMA_STORE{},tensor(o),typename T::SmemLayoutO{},Shape<_128,_64>{},_1{});
    auto f=ffpa_attn_tma_split_d_cute<T,decltype(tq),decltype(tk),decltype(tv),decltype(to)>;
    int bytes=T::kSmemElems*sizeof(E);
    C10_CUDA_CHECK(cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    f<<<dim3(nq/128,bh),T::kNumThreads,bytes,stream>>>(tq,tk,tv,to,ptr(o),lse.data_ptr<float>(),bh*nq,bh*nk,nq,nk);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<int SQ,int SV>
void variant(Tensor q,Tensor k,Tensor v,Tensor o,Tensor lse,int kind) {
  if(kind==0)launch<FFPA_HEAD_DIM,SQ,SV,0>(q,k,v,o,lse);
  else if(kind==1)launch<FFPA_HEAD_DIM,SQ,SV,1>(q,k,v,o,lse);
  else launch<FFPA_HEAD_DIM,SQ,SV,2>(q,k,v,o,lse);
}
void forward(Tensor q,Tensor k,Tensor v,Tensor o,Tensor lse,int kind,int sq,int sv) {
  c10::cuda::CUDAGuard guard(q.device());
  TORCH_CHECK(q.size(3)==FFPA_HEAD_DIM,"D mismatch");
#define S(Q,V) if(sq==Q&&sv==V){variant<Q,V>(q,k,v,o,lse,kind);return;}
  S(1,1);S(2,2);S(2,3);S(3,2);S(3,3);
#undef S
  TORCH_CHECK(false,"CUDA reference stage pair not instantiated");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
