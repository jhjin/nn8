#include "TH.h"
#include "luaT.h"

#include "lib/qgemm_wrapper.h"

#include "lib/tensor.c"
#include "lib/vector.c"

#include "generic/Threshold.c"
#include "generic/SpatialConvolutionMM.c"
#include "generic/SpatialMaxPooling.c"


LUA_EXTERNC DLL_EXPORT int luaopen_libnn8(lua_State *L);

int luaopen_libnn8(lua_State *L)
{
  lua_newtable(L);

  nn8_ByteThreshold_init(L);
  nn8_ByteSpatialConvolutionMM_init(L);
  nn8_ByteSpatialMaxPooling_init(L);

  return 1;
}
