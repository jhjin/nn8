#ifndef QGEMM_H
#define QGEMM_H


#ifdef __cplusplus
#include <cstdint>

#include "gemmlowp/eight_bit_int_gemm/eight_bit_int_gemm.h"
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
