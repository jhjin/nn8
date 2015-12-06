require('nn8')
torch.manualSeed(2)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)
print('==> #threads: ', torch.getnumthreads())


local batchSize = 128
local iC = 3
local iH = 224
local iW = iH


-- alex krizhevsky one weird trick (http://arxiv.org/abs/1404.5997)
local model32 = nn.Sequential()
model32:add(nn.SpatialConvolutionMM(3,64,11,11,4,4,2,2))
model32:add(nn.Threshold(10, 10))
model32:add(nn.SpatialMaxPooling(3,3,2,2))
model32:add(nn.SpatialConvolutionMM(64,192,5,5,1,1,2,2))
model32:add(nn.Threshold(10, 10))
model32:add(nn.SpatialMaxPooling(3,3,2,2))
model32:add(nn.SpatialConvolutionMM(192,384,3,3,1,1,1,1))
model32:add(nn.Threshold(10, 10))
model32:add(nn.SpatialConvolutionMM(384,256,3,3,1,1,1,1))
model32:add(nn.Threshold(10, 10))
model32:add(nn.SpatialConvolutionMM(256,256,3,3,1,1,1,1))
model32:add(nn.Threshold(10, 10))
model32:add(nn.SpatialMaxPooling(3,3,2,2))
model32:add(nn.View(256*6*6))

--[[ modules not supported yet
model32:add(nn.Linear(256*6*6, 4096))
model32:add(nn.Threshold(10, 10))
model32:add(nn.Linear(4096, 4096))
model32:add(nn.Threshold(10, 10))
model32:add(nn.Linear(4096, 1000))
model32:add(nn.SoftMax())
]]

for _, v in pairs({1,4,7,9,11}) do
   model32:get(v).weight:random(3):add(-1)
   model32:get(v).bias:random(3):add(-1)
end


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
