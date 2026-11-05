#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// CuTe local_tile(idx,8) assigns eight adjacent half values to each thread.
// Preserve the two nested half2 operations and the 128-bit vector transfer.
template<int Elements=8>
__global__ void vector_add_local_tile_multi_elem_per_thread_half(
    half *z,int num,const half *x,const half *y,half a,half b,half c) {
  static_assert(Elements>0 && Elements%2==0);
  int start=(blockIdx.x*blockDim.x+threadIdx.x)*Elements;
  if(start>=num)return;
  half2 aa=__half2half2(a),bb=__half2half2(b),cc=__half2half2(c);
  if(start+Elements<=num && (uintptr_t(x+start)|uintptr_t(y+start)|uintptr_t(z+start))%16==0 && Elements%8==0){
#pragma unroll
    for(int block=0;block<Elements;block+=8){
      union alignas(16) Pack { uint4 bits;half2 h[4]; } px,py,pz;
      px.bits=*reinterpret_cast<const uint4*>(x+start+block);
      py.bits=*reinterpret_cast<const uint4*>(y+start+block);
#pragma unroll
      for(int i=0;i<4;i++)pz.h[i]=__hadd2(__hmul2(px.h[i],aa),__hadd2(__hmul2(py.h[i],bb),cc));
      *reinterpret_cast<uint4*>(z+start+block)=pz.bits;
    }
  }else{
#pragma unroll
    for(int i=0;i<Elements;i++)if(start+i<num)z[start+i]=__hadd(__hmul(x[start+i],a),__hadd(__hmul(y[start+i],b),c));
  }
}
