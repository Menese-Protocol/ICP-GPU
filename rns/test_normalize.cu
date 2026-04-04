#include <cstdio>
#include <cstdint>
#include "rns_fp_v2.cuh"
__global__ void test() {
    if (threadIdx.x != 0) return;
    RnsFp one = rns_one();
    RnsFp zero = rns_zero();
    RnsFp one_plus_p = rns_sub(one, zero);  // one - 0 + p = one + p
    
    printf("one.rr = %u\n", one.rr);
    printf("one+p.rr = %u\n", one_plus_p.rr);
    
    RnsFp norm = rns_normalize(one_plus_p);
    printf("normalize(one+p).rr = %u\n", norm.rr);
    printf("match one: %s\n", rns_eq_raw(norm, one)?"YES":"NO");
    
    // Also test: normalize(one) = one
    RnsFp norm_one = rns_normalize(one);
    printf("normalize(one).rr = %u\n", norm_one.rr);
    printf("norm(one)==one: %s\n", rns_eq_raw(norm_one, one)?"YES":"NO");
}
int main() { test<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
