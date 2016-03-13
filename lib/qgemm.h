#ifndef QGEMM_H
#define QGEMM_H


#ifdef __cplusplus
#include <cstdint>
#include <tuple>

#include "gemmlowp/public/gemmlowp.h"

extern "C" {
#endif

void THByteBlas_gemm8(uint8_t* c, uint8_t* c_bias,
                      const uint8_t* a, const uint8_t* b,
                      const int m, const int n, const int k,
                      const int a_offset, const int b_offset,
                      const int c_offset, const int c_mult, const int c_shift,
                      const int use_relu);

#ifdef __cplusplus
}
#endif

#endif
