#pragma once
#include <hip/hip_runtime.h>
namespace leetcuda_transpose_examples {
template<int Mode>__global__ void transpose(const int *a,int *b,int m,int n){
  __shared__ int tile[32][Mode==1?33:32];
  int x=threadIdx.x,y=threadIdx.y;
#pragma unroll
  for(int r=y;r<32;r+=8){
    int row=blockIdx.y*32+r,col=blockIdx.x*32+x;
    tile[x][Mode==2?(x^r):r]=(row<m && col<n)?a[size_t(row)*n+col]:0;
  }
  __syncthreads();
#pragma unroll
  for(int r=y;r<32;r+=8){
    int row=blockIdx.x*32+r,col=blockIdx.y*32+x;
    if(row<n && col<m)b[size_t(row)*m+col]=tile[r][Mode==2?(r^x):x];
  }
}
}
