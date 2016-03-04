require('torch')
require('nn')
require('THNN')

-- temporary support for byte copy for nn.Module class
function nn.Module:byte()
   return self:type('torch.ByteTensor')
end


return nn
