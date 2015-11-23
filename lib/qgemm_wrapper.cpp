#include "qgemm_wrapper.h"
#include "qgemm.h"

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
                      const int c_mult, const int c_shift) {

  qgemm_(is_a_transposed, is_b_transposed, is_c_transposed,
         m, n, k, a, b, c, lda, ldb, ldc,
         a_offset, b_offset, c_offset, c_mult, c_shift);
}

#ifdef __cplusplus
}
#endif
