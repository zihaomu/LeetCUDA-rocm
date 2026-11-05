#include "bench_runtime.hpp"
#include "profile_compute.hpp"
#include "cute_vector_hip.cuh"
#include "transpose_examples_hip.cuh"
#include "cute_dsl_gemm_hip.cuh"
#include "swizzle_examples_hip.cuh"
using namespace leetcuda_bench;

template<class T,class Fn>void check_output(Harness&h,const std::string&id,const std::vector<double>&ref,Fn fn,double atol=0,double rtol=0){
 Buffer<T>o(ref.size()+16);std::vector<T>initial(o.n);for(size_t i=0;i<o.n;i++)initial[i]=T(-171);for(int i=0;i<8;i++)initial[i]=initial[o.n-1-i]=T(91);
 auto check_values=[&](std::string suffix){auto all=o.download();std::vector<double>x(ref.size());bool canary=true;for(int i=0;i<8;i++)canary &= double(all[i])==91 && double(all[o.n-1-i])==91;for(size_t i=0;i<x.size();i++)x[i]=double(all[i+8]);h.record(id+suffix,x,ref,atol,rtol,canary);};
 o.upload(initial);fn(o.p+8);check(hipGetLastError());check(hipStreamSynchronize(h.stream));check_values("");
 hipGraph_t graph;hipGraphExec_t exec;check(hipStreamBeginCapture(h.stream,hipStreamCaptureModeGlobal));fn(o.p+8);check(hipStreamEndCapture(h.stream,&graph));check(hipGraphInstantiateWithFlags(&exec,graph,0));o.upload(initial);check(hipGraphLaunch(exec,h.stream));check(hipStreamSynchronize(h.stream));check_values("/graph");check(hipGraphExecDestroy(exec));check(hipGraphDestroy(graph));
}
void profiles(Harness&h,const std::vector<float>&f){
 constexpr int n=8192;Buffer<float>a(n),b(n);Buffer<_Float16>ha(n),hb(n);std::vector<float>x(n),y(n);std::vector<_Float16>hx(n),hy(n);std::vector<double>add(n),relu(n),hadd(n),hrelu(n);
 for(int i=0;i<n;i++){x[i]=f[i%f.size()];y[i]=f[(i+8192)%f.size()];hx[i]=_Float16(x[i]);hy[i]=_Float16(y[i]);add[i]=float(x[i]+y[i]);relu[i]=std::max(x[i],0.f);hadd[i]=double(_Float16(double(hx[i])+double(hy[i])));hrelu[i]=std::max(double(hx[i]),0.);}
 a.upload(x);b.upload(y);ha.upload(hx);hb.upload(hy);
#define FADD(F,W) check_output<float>(h,#F,add,[&](float*o){F<<<n/(256*W),256,0,h.stream>>>(a.p,b.p,o,n);})
#define FRELU(F,W) check_output<float>(h,#F,relu,[&](float*o){F<<<n/(256*W),256,0,h.stream>>>(a.p,o,n);})
#define HADD(F,W) check_output<_Float16>(h,#F,hadd,[&](_Float16*o){F<<<n/(256*W),256,0,h.stream>>>((half*)ha.p,(half*)hb.p,(half*)o,n);})
#define HRELU(F,W) check_output<_Float16>(h,#F,hrelu,[&](_Float16*o){F<<<n/(256*W),256,0,h.stream>>>((half*)ha.p,(half*)o,n);})
 FADD(elementwise_add_f32_kernel,1);FADD(elementwise_add_f32x4_kernel,4);HADD(elementwise_add_f16_kernel,1);HADD(elementwise_add_f16x2_kernel,2);HADD(elementwise_add_f16x8_kernel,8);HADD(elementwise_add_f16x8_pack_kernel,8);
 FRELU(relu_f32_kernel,1);FRELU(relu_f32x4_kernel,4);HRELU(relu_f16_kernel,1);HRELU(relu_f16x2_kernel,2);HRELU(relu_f16x8_kernel,8);HRELU(relu_f16x8_pack_kernel,8);
#undef FADD
#undef FRELU
#undef HADD
#undef HRELU
}
template<int E>void vector(Harness&h,const std::vector<float>&f){
 for(int n:{17,8192}){
  Buffer<_Float16>a(n),b(n);std::vector<_Float16>x(n),y(n);std::vector<double>ref(n);_Float16 aa=_Float16(.3),bb=_Float16(-.7),cc=_Float16(.13);
  for(int i=0;i<n;i++){x[i]=_Float16(f[i%f.size()]);y[i]=_Float16(f[(i+8192)%f.size()]);ref[i]=double(_Float16(double(x[i])*double(aa)+double(_Float16(double(y[i])*double(bb)+double(cc)))));}
  a.upload(x);b.upload(y);check_output<_Float16>(h,"vector/"+std::to_string(E)+"/"+std::to_string(n),ref,[&](_Float16*o){vector_add_local_tile_multi_elem_per_thread_half<E><<<(n+256*E-1)/(256*E),256,0,h.stream>>>((half*)o,n,(half*)a.p,(half*)b.p,__float2half(float(aa)),__float2half(float(bb)),__float2half(float(cc)));});
 }
}
template<int Mode>void transpose(Harness&h){
 for(auto shape:{std::vector<int>{32,64},{17,137}}){int m=shape[0],n=shape[1];Buffer<int>a(m*n);std::vector<int>x(a.n);std::vector<double>ref(a.n);for(int r=0;r<m;r++)for(int c=0;c<n;c++)ref[c*m+r]=x[r*n+c]=r*n+c-73;a.upload(x);check_output<int>(h,"transpose/"+std::to_string(Mode)+"/"+std::to_string(m),ref,[&](int*o){leetcuda_transpose_examples::transpose<Mode><<<dim3((n+31)/32,(m+31)/32),dim3(32,8),0,h.stream>>>(a.p,o,m,n);});}
}
template<int Kind>void dsl(Harness&h,const std::vector<float>&f){
 for(auto shape:{std::vector<int>{128,128,64},{17,33,65}}){int m=shape[0],n=shape[1],k=shape[2];Buffer<float>a(m*k),b(k*n);std::vector<float>x(a.n),y(b.n);std::vector<double>ref(m*n);
  for(int i=0;i<m*k;i++)x[i]=f[i%f.size()]*.5f;for(int i=0;i<k;i++)for(int j=0;j<n;j++)y[Kind==0?i*n+j:j*k+i]=f[(i*n+j+8192)%f.size()]*.5f;
  for(int i=0;i<m;i++)for(int j=0;j<n;j++){double sum=0;for(int t=0;t<k;t++)sum+=double(x[i*k+t])*y[Kind==0?t*n+j:j*k+t];ref[i*n+j]=Kind==0?double(_Float16(sum)):sum;}
  a.upload(x);b.upload(y);check_output<float>(h,"dsl/"+std::to_string(Kind)+"/"+std::to_string(m),ref,[&](float*o){if constexpr(Kind==0)leetcuda_cute_dsl::naive<<<dim3((n+31)/32,(m+31)/32),dim3(32,32),0,h.stream>>>(a.p,b.p,o,m,n,k);else leetcuda_cute_dsl::tiled<Kind==2><<<dim3((m+127)/128,(n+127)/128),256,0,h.stream>>>(a.p,b.p,o,m,n,k,nullptr);},Kind==0?.01:.0002,Kind==0?.01:.0002);
 }
}
template<int Kind,int AP=0,int BP=0>void swizzle(Harness&h,const std::vector<float>&f){
 for(auto shape:{std::vector<int>{128,128,64},{17,33,65}}){int m=shape[0],n=shape[1],k=shape[2];Buffer<_Float16>a(m*k),b(k*n);std::vector<_Float16>x(a.n),y(b.n);std::vector<double>ref(m*n);
  for(int i=0;i<m*k;i++)x[i]=_Float16(f[i%f.size()]*.25f);for(int i=0;i<k*n;i++)y[i]=_Float16(f[(i+8192)%f.size()]*.25f);
  for(int i=0;i<m;i++)for(int j=0;j<n;j++){double sum=0;for(int t=0;t<k;t++)sum+=double(x[i*k+t])*double(y[t*n+j]);ref[i*n+j]=double(_Float16(sum));}a.upload(x);b.upload(y);
  constexpr bool large=Kind==1||Kind==3;constexpr int bm=large?128:16,bn=large?128:8;
  check_output<_Float16>(h,"swizzle/"+std::to_string(Kind)+"/"+std::to_string(AP)+"/"+std::to_string(BP)+"/"+std::to_string(m),ref,[&](_Float16*o){leetcuda_swizzle_examples::gemm<Kind,AP,BP><<<dim3((n+bn-1)/bn,(m+bm-1)/bm),large?256:32,0,h.stream>>>(a.p,b.p,o,m,n,k);},.01,0);
 }
}
int main(int argc,char**argv){try{if(argc!=3)throw std::runtime_error("usage: supplemental_bench fixture.f32 output.f64");Harness h(argv[2],false);auto f=read_fixture(argv[1]);profiles(h,f);vector<8>(h,f);vector<16>(h,f);transpose<0>(h);transpose<1>(h);transpose<2>(h);dsl<0>(h,f);dsl<1>(h,f);dsl<2>(h,f);swizzle<0>(h,f);swizzle<2>(h,f);swizzle<4>(h,f);
#define P(K) swizzle<K,0,0>(h,f);swizzle<K,0,8>(h,f);swizzle<K,8,0>(h,f);swizzle<K,8,8>(h,f)
P(1);P(3);
#undef P
return h.passed?0:1;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
