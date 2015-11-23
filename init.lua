require('torch')
require('nn')
require('libnn8')

include('transform.lua')


-- temporary support for byte copy for nn.Module class
function nn.Module:byte()
   return self:type('torch.ByteTensor')
end


return nn
