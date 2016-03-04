local ffi = require 'ffi'

local THNN = {}

local generic_THNN_h = [[
TH_API void THNN_(Threshold_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          real threshold,
          real val,
          bool inplace);

TH_API void THNN_(SpatialConvolutionMM_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH);

TH_API void THNN_(SpatialMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *indices,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode);
]]

-- THGenerator struct declaration copied from torch7/lib/TH/THRandom.h
local base_declarations = [[
typedef void THNNState;

typedef struct {
  unsigned long the_initial_seed;
  int left;
  int seeded;
  unsigned long next;
  unsigned long state[624]; /* the array for the state vector 624 = _MERSENNE_STATE_N  */
  double normal_x;
  double normal_y;
  double normal_rho;
  int normal_is_valid;
} THGenerator;
]]

-- polyfill for LUA 5.1
if not package.searchpath then
   local sep = package.config:sub(1,1)
   function package.searchpath(mod, path)
      mod = mod:gsub('%.', sep)
      for m in path:gmatch('[^;]+') do
         local nm = m:gsub('?', mod)
         local f = io.open(nm, 'r')
         if f then
            f:close()
            return nm
         end
     end
   end
end

-- load libTHNN
THNN.C = ffi.load(package.searchpath('libnn8', package.cpath))

ffi.cdef(base_declarations)

-- expand macros, allow to use original lines from lib/THNN/generic/THNN.h
local preprocessed = string.gsub(generic_THNN_h, 'TH_API void THNN_%(([%a%d_]+)%)', 'void THNN_TYPE%1')

local replacements =
{
   {
      ['TYPE'] = 'Byte',
      ['real'] = 'uint8_t',
      ['THTensor'] = 'THByteTensor',
      ['THIndexTensor'] = 'THLongTensor',
      ['THIntegerTensor'] = 'THIntTensor',
      ['THIndex_t'] = 'long',
      ['THInteger_t'] = 'int'
   },
}

for i=1,#replacements do
   local r = replacements[i]
   local s = preprocessed
   for k,v in pairs(r) do
      s = string.gsub(s, k, v)
   end
   ffi.cdef(s)
end

THNN.NULL = ffi.NULL or nil

function THNN.getState()
   return ffi.NULL or nil
end

function THNN.optionalTensor(t)
   return t and t:cdata() or THNN.NULL
end

local function extract_function_names(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API void THNN_%(([%a%d_]+)%)') do
      t[#t+1] = n
   end
   return t
end

function THNN.bind(lib, base_names, type_name, state_getter)
   local ftable = {}
   local prefix = 'THNN_' .. type_name
   for i,n in ipairs(base_names) do
      -- use pcall since some libs might not support all functions (e.g. cunn)
      local ok,v = pcall(function() return lib[prefix .. n] end)
      if ok then
         ftable[n] = function(...) v(state_getter(), ...) end   -- implicitely add state
      else
         print('not found: ' .. prefix .. n .. v)
      end
   end
   return ftable
end

-- build function table
local function_names = extract_function_names(generic_THNN_h)

THNN.kernels = {}
THNN.kernels['torch.ByteTensor'] = THNN.bind(THNN.C, function_names, 'Byte', THNN.getState)

torch.getmetatable('torch.ByteTensor').THNN = THNN.kernels['torch.ByteTensor']

function THNN.runKernel(f, type, ...)
   local ftable = THNN.kernels[type]
   if not ftable then
      error('Unsupported tensor type: '..type)
   end
   local f = ftable[f]
   if not f then
      error(string.format("Function '%s' not found for tensor type '%s'.", f, type))
   end
   f(...)
end

return THNN
