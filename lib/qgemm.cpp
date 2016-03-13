#include "qgemm.h"

void THByteBlas_gemm8(uint8_t* c, uint8_t* c_bias,
                      const uint8_t* a, const uint8_t* b,
                      const int m, const int n, const int k,
                      const int a_offset, const int b_offset,
                      const int c_offset, const int c_mult, const int c_shift,
                      const int use_relu)
{
  // quantize-down, unclamped (but scaled) int32's
  gemmlowp::OutputStageQuantizeDownInt32ToUint8Scale quantize_down_stage;
  quantize_down_stage.result_offset = c_offset;
  quantize_down_stage.result_mult_int = c_mult;
  quantize_down_stage.result_shift = c_shift;

  // clamp-and-cast-to-uint8
  gemmlowp::OutputStageSaturatingCastToUint8 saturating_cast_stage;

  // clamp min/max bounds
  gemmlowp::OutputStageClamp clamp_stage{0, 255};
  if (use_relu) clamp_stage.min = 128;

  typedef gemmlowp::VectorMap<std::uint8_t, gemmlowp::VectorShape::Col> ColVectorMap;
  ColVectorMap col_vector_map(c_bias, m);
  gemmlowp::OutputStageBiasAddition<ColVectorMap> col_bias_addition_stage;
  col_bias_addition_stage.bias_vector = col_vector_map;

  // set pipeline after gemm
  auto bias_clamp_quantize_cast_pipeline =
      std::make_tuple(col_bias_addition_stage,
                      clamp_stage,
                      quantize_down_stage,
                      saturating_cast_stage);

  // init gemmlowp context and storage
  gemmlowp::GemmContext context;
  const gemmlowp::MatrixMap<const std::uint8_t, gemmlowp::MapOrder::RowMajor> a_(a, m, k, k);
  const gemmlowp::MatrixMap<const std::uint8_t, gemmlowp::MapOrder::RowMajor> b_(b, k, n, n);
  gemmlowp::MatrixMap<std::uint8_t, gemmlowp::MapOrder::RowMajor> c_(c, m, n, n);

  // gemm and output pipeline
  gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::uint8_t, gemmlowp::DefaultL8R8BitDepthParams>(
      &context, a_, b_, &c_, a_offset, b_offset, bias_clamp_quantize_cast_pipeline);
}
