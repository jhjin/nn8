#include "luaT.h"

#include <stdint.h>
#include <stdbool.h>
#include <TH.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef void THNNState;

TH_API void THNN_ByteThreshold_updateOutput(
          THNNState *state,
          THByteTensor *input,
          THByteTensor *output,
          uint8_t threshold,
          uint8_t val,
          bool inplace);

TH_API void THNN_ByteSpatialConvolutionMM_updateOutput(
          THNNState *state,
          THByteTensor *input,
          THByteTensor *output,
          THByteTensor *weight,
          THByteTensor *bias,
          THByteTensor *finput,
          THByteTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH);

TH_API void THNN_ByteSpatialMaxPooling_updateOutput(
          THNNState *state,
          THByteTensor *input,
          THByteTensor *output,
          THByteTensor *indices,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode);

#include "lib/qgemm_wrapper.h"

#include "lib/tensor.c"
#include "lib/vector.c"

#include "generic/Threshold.c"
#include "generic/SpatialConvolutionMM.c"
#include "generic/SpatialMaxPooling.c"
