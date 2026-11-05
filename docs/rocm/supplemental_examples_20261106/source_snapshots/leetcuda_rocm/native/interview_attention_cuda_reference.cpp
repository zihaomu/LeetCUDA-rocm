// Test-only launcher for immutable upstream interview attention definitions.
// No kernel arithmetic adaptation: manual precision is selected by the original
// kMmaAccF32 template. Current-stream and descriptor lifetime are explicit.
#include "hgemm.cuh"
#include "flash_attn.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
using namespace cute;
namespace {
using torch::Tensor;
constexpr int Kind=INTERVIEW_KIND,D=INTERVIEW_D,Acc=INTERVIEW_ACC_F32;
half* ptr(Tensor t){return reinterpret_cast<half*>(t.data_ptr());}
cutlass::half_t* cptr(Tensor t){return reinterpret_cast<cutlass::half_t*>(t.data_ptr());}
template<class Kernel>void configure(Kernel fn,int bytes){
 int device,limit;cudaFuncAttributes attr{};C10_CUDA_CHECK(cudaGetDevice(&device));C10_CUDA_CHECK(cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,fn));
 TORCH_CHECK(bytes+attr.sharedSizeBytes<=size_t(limit),"original CUDA shared-memory requirement ",bytes+attr.sharedSizeBytes," exceeds ",limit);
 C10_CUDA_CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
}
Tensor prepare(Tensor q,Tensor k,Tensor v){
 c10::cuda::CUDAGuard guard(q.device());
 auto cpu=torch::empty({3,int64_t(sizeof(CUtensorMap))},torch::dtype(torch::kUInt8).device(torch::kCPU));
 if constexpr(Kind==1 || Kind==2){
  alignas(128) CUtensorMap maps[3];
  Tensor inputs[]={q,k,v};constexpr int br=Kind==1?128:64;
  for(int i=0;i<3;i++){
   uint64_t shape[2]={D,uint64_t(inputs[i].numel()/D)},stride[1]={D*sizeof(half)};
   uint32_t box[2]={64,uint32_t(i==0?br:64)},step[2]={1,1};
   CUresult result=cuTensorMapEncodeTiled(maps+i,CU_TENSOR_MAP_DATA_TYPE_FLOAT16,2,ptr(inputs[i]),shape,stride,box,step,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
   TORCH_CHECK(result==CUDA_SUCCESS,"CUDA tensor map encode failed ",int(result));
  }
  std::memcpy(cpu.data_ptr(),maps,sizeof(maps));
 }else cpu.zero_();
 return cpu.to(q.device());
}
template<int SK,int SV,int Pad>
void launch(Tensor q,Tensor k,Tensor v,Tensor o,Tensor maps,bool causal,float scale){
 int nq=q.size(-2),nk=k.size(-2),bh=q.numel()/(nq*D);auto stream=at::cuda::getCurrentCUDAStream();
 if constexpr(Kind==0){
  TORCH_CHECK(nq==nk && nq%128==0,"original basic requires N%128=0 self attention");
  auto f=flash_attn_mma_stages_split_q<D,16,8,16,Acc,8,1,8,1,1,8,1,D/8,SK,Pad,Pad,Pad>;
  int bytes=(128+(SK+1)*64)*(D+Pad)*2;configure(f,bytes);
  f<<<dim3(nq/128,bh),256,bytes,stream>>>(ptr(q),ptr(k),ptr(v),ptr(o),nq,q.size(1));
 }else if constexpr(Kind==1 || Kind==2){
  constexpr int br=Kind==1?128:64,groups=Kind==1?1:2;
  TORCH_CHECK(nq==nk && nq%br==0,"original WS requires aligned self attention");
  int bytes=(br+groups*(SK+SV)*64)*D*2;
  auto tq=reinterpret_cast<const CUtensorMap*>(maps.data_ptr());
  if constexpr(Kind==1){
   auto f=flash_attn_tma_mma_ws_stages_split_q<D,16,8,16,Acc,8,1,8,1,1,8,1,D/8,SK,SV,384>;configure(f,bytes);
   f<<<dim3(nq/br,bh),384,bytes,stream>>>(ptr(q),ptr(k),ptr(v),ptr(o),nq,q.size(1),tq,tq+1,tq+2);
  }else{
   static_assert(SV==1);auto f=flash_attn_3_tma_ws_stages_split_q<D,16,8,16,Acc,4,1,4,1,1,8,1,D/8,SK,SV,384>;configure(f,bytes);
   f<<<dim3(nq/br,bh),384,bytes,stream>>>(ptr(q),ptr(k),ptr(v),ptr(o),nq,q.size(1),tq,tq+1,tq+2);
  }
 }else if constexpr(Kind==3){
  TORCH_CHECK(nq==nk && nq%128==0,"original CuTe requires N%128=0 self attention");
  auto f=flash_attn_mma_stages_split_q_cute<D,SK>;int bytes=(128+(SK+1)*64)*D*2;configure(f,bytes);
  f<<<dim3(nq/128,bh),256,bytes,stream>>>(cptr(q),cptr(k),cptr(v),cptr(o),bh*nq,nq);
 }else if constexpr(Kind==4 || Kind==5){
  constexpr int br=Kind==4?128:64;
  TORCH_CHECK(nq==nk && nq%br==0,"original CuTe WS requires aligned self attention");
  using T=std::conditional_t<Kind==4,fa_cute::FlashAttn2CuTeTraits<D>,fa_cute::FlashAttn3CuTeTraits<D>>;
  // Both traits expose an equivalent Q/K/V swizzle atom; use explicit shape.
  using LQ=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<cutlass::half_t>{},Shape<Int<br>,Int<D>>{}));
  using LKV=decltype(tile_to_shape(GMMA::Layout_K_SW128_Atom<cutlass::half_t>{},Shape<_64,Int<D>>{}));
  auto tensor=[&](Tensor x){return make_tensor(make_gmem_ptr(cptr(x)),make_shape(bh*nq,Int<D>{}),make_stride(Int<D>{},_1{}));};
  auto tq=make_tma_copy(SM90_TMA_LOAD{},tensor(q),LQ{},Shape<Int<br>,Int<D>>{},_1{});
  auto tk=make_tma_copy(SM90_TMA_LOAD{},tensor(k),LKV{},Shape<_64,Int<D>>{},_1{});
  auto tv=make_tma_copy(SM90_TMA_LOAD{},tensor(v),LKV{},Shape<_64,Int<D>>{},_1{});
  int bytes=(br+(Kind==4?1:2)*(SK+SV)*64)*D*2;
  if constexpr(Kind==4){auto f=flash_attn_tma_mma_ws_split_q_cute<D,decltype(tq),decltype(tk),decltype(tv),SK,SV>;configure(f,bytes);f<<<dim3(nq/br,bh),384,bytes,stream>>>(tq,tk,tv,cptr(o),bh*nq,nq);}
  else {static_assert(SV==1);auto f=flash_attn_3_tma_mma_ws_split_q_cute<D,decltype(tq),decltype(tk),decltype(tv),SK>;configure(f,bytes);f<<<dim3(nq/br,bh),384,bytes,stream>>>(tq,tk,tv,cptr(o),bh*nq,nq);}
 }else if constexpr(Kind==7){
  // Preserve the exact original persist-D traits and current-stream host path.
  TORCH_CHECK(SK==2 && SV==2,"original persist-D convenience entry uses depth 2");
  flash_attn_cute_persist_d_sm120_fwd(cptr(q),cptr(k),cptr(v),cptr(o),q.size(0),q.size(1),k.size(1),nq,nk,D,causal,scale,stream);
 }
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void interview_reference_forward(Tensor q,Tensor k,Tensor v,Tensor o,Tensor maps,int sk,int sv,int padmask,bool causal,float scale){
 c10::cuda::CUDAGuard guard(q.device());TORCH_CHECK(q.size(-1)==D,"CUDA reference D mismatch");
 if constexpr(Kind==0){
  TORCH_CHECK(sv==1 && (padmask==0 || padmask==7),"CUDA basic reference supports pad all0/all8");
#define S(K) if(sk==K){if(padmask==0)launch<K,1,0>(q,k,v,o,maps,causal,scale);else launch<K,1,8>(q,k,v,o,maps,causal,scale);return;}
  S(1);S(2);S(3);S(4);
#undef S
 }else if constexpr(Kind==1 || Kind==4){
#define S(K,V) if(sk==K && sv==V){launch<K,V,0>(q,k,v,o,maps,causal,scale);return;}
  S(1,1);S(1,2);S(2,1);S(2,2);S(3,1);S(3,2);S(4,1);S(4,2);
#undef S
 }else if constexpr(Kind==7){launch<2,2,0>(q,k,v,o,maps,causal,scale);return;}
 else if constexpr(Kind==6){
  int rows=q.numel()/D;TORCH_CHECK(rows%64==0,"original copy requires rows%64=0");
  using L=typename fa_cute::FlashAttn3CuTeTraits<D>::SmemLayoutQKV;
  auto tensor=make_tensor(make_gmem_ptr(cptr(q)),make_shape(rows,Int<D>{}),make_stride(Int<D>{},_1{}));
  auto tq=make_tma_copy(SM90_TMA_LOAD{},tensor,L{},Shape<_64,Int<D>>{},_1{});
  auto f=flash_attn_3_cute_tma_copy_smoke<D,decltype(tq)>;int bytes=cosize(L{})*2;configure(f,bytes);
  f<<<rows/64,128,bytes,at::cuda::getCurrentCUDAStream()>>>(tq,cptr(o),rows);C10_CUDA_KERNEL_LAUNCH_CHECK();return;
 }else{
  TORCH_CHECK(sv==1,"single V slot required");
#define S(K) if(sk==K){launch<K,1,0>(q,k,v,o,maps,causal,scale);return;}
  S(1);S(2);S(3);S(4);
#undef S
 }
 TORCH_CHECK(false,"unsupported CUDA stage");
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("prepare",&prepare);m.def("forward",&interview_reference_forward);}
