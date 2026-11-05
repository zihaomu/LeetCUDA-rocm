#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/pointer_flagged.hpp>
#include <cutlass/half.h>
#include <cstdio>
using namespace cute;
template<int D,int Rows>void dump(){
 using Atom=std::conditional_t<D==96,GMMA::Layout_K_SW64_Atom<cutlass::half_t>,GMMA::Layout_K_SW128_Atom<cutlass::half_t>>;
 auto original=tile_to_shape(Atom{},Shape<Int<Rows>,Int<D>>{});
 auto layout=as_position_independent_swizzle_layout(original);
 printf("{\"d\":%d,\"rows\":%d,\"cosize\":%d,\"offsets\":[",D,Rows,int(cosize(layout)));
 for(int row=0;row<Rows;row++)for(int col=0;col<D;col++)printf("%s%d",row||col?",":"",int(layout(make_coord(row,col))));
 printf("]}\n");
}
int main(){dump<64,16>();dump<64,32>();dump<64,64>();dump<96,16>();dump<96,32>();dump<96,64>();dump<128,16>();dump<128,32>();dump<128,64>();}
