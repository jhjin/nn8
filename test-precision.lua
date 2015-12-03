require('nn8')
torch.manualSeed(2)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)
print('==> #threads: ', torch.getnumthreads())


local batchSize = 128
local iC = 3
local iH = 224
local iW = iH


local model32 = nn.Sequential()
model32:add(nn.SpatialConvolutionMM(iC, 8, 3, 3))
model32:add(nn.SpatialMaxPooling(2, 2, 2, 2))
model32:add(nn.Threshold(10, 10))
model32:get(1).weight:random(3):add(-1)
model32:get(1).bias:random(3):add(-1)


local model8 = nn.Sequential()
for i = 1, #model32 do
   model8:add(model32:get(i):clone())
end
model8:byte()


local x32 = torch.FloatTensor(batchSize,iC,iH,iW):random(3):add(-1)
local x8  = x32:byte()


local t32 = torch.Timer()
local y32 = model32:forward(x32)
print('==> 32-bit: ', t32:time().real)


local t8  = torch.Timer()
local y8  = model8:forward(x8)
print('==>  8-bit: ', t8:time().real)


local diff = (y32 - y8:float()):abs()
print('==> diff [max]: ', diff:max())
