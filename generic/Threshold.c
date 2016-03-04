void THNN_ByteThreshold_updateOutput(
          THNNState *state,
          THByteTensor *input,
          THByteTensor *output,
          uint8_t threshold,
          uint8_t val,
          bool inplace)
{
  if (inplace) {
    TH_TENSOR_APPLY(uint8_t, input,
      if (*input_data <= threshold)
        *input_data = val;
    );
    THByteTensor_set(output, input);
  } else {
    THByteTensor_resizeAs(output, input);
    TH_TENSOR_APPLY2(uint8_t, output, uint8_t, input,
      *output_data = (*input_data > threshold) ? *input_data : val;
    );
  }
}
