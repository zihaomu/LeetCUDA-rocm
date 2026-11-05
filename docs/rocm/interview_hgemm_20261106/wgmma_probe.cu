#define NOTES_V2_ENABLE_WGMMA 1
#include "interview/hgemm.cuh"
void instantiate(half *c, const CUtensorMap *a, const CUtensorMap *b) {
 hgemm_wgmma_stages_tn<><<<1,256,98304>>>(128,128,64,c,a,b);
}
