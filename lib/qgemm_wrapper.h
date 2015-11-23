#ifndef QGEMM_WRAPPER_H 
#define QGEMM_WRAPPER_H 

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void THByteBlas_gemm8(const int is_a_transposed,
                      const int is_b_transposed,
                      const int is_c_transposed,
                      const int m, const int n, const int k,
                      const uint8_t* a, const uint8_t* b, uint8_t* c,
                      const int lda, const int ldb, const int ldc,
                      const int a_offset, const int b_offset, const int c_offset,
                      const int c_mult, const int c_shift);

#ifdef __cplusplus
}
#endif

#endif
