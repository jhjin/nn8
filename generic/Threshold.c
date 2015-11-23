static int nn8_ByteThreshold_updateOutput(lua_State *L)
{
  THByteTensor *input = luaT_checkudata(L, 2, "torch.ByteTensor");
  uint8_t val = (uint8_t) luaT_getfieldchecknumber(L, 1, "val");
  uint8_t threshold = (uint8_t) luaT_getfieldchecknumber(L, 1, "threshold");
  THByteTensor *output = luaT_getfieldcheckudata(L, 1, "output", "torch.ByteTensor");
  int inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");


  if (inPlace) {
    TH_TENSOR_APPLY(uint8_t, input,                  \
                    if (*input_data <= threshold) {  \
                      *input_data = val;             \
                    });
    THByteTensor_set(output, input);
  } else {
    THByteTensor_resizeAs(output, input);
    TH_TENSOR_APPLY2(uint8_t, output, uint8_t, input,                                \
                     *output_data = (*input_data > threshold) ? *input_data : val;);
  }

  return 1;
}

static const struct luaL_Reg nn8_ByteThreshold__ [] = {
  {"Threshold_updateOutput", nn8_ByteThreshold_updateOutput},
  {NULL, NULL}
};

static void nn8_ByteThreshold_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.ByteTensor");
  luaT_registeratname(L, nn8_ByteThreshold__, "nn");
  lua_pop(L,1);
}
