-- [SublimeLinter luacheck-globals:+cudnn,deepwarp]

require 'cudnn'
require 'cunn'
-- require 'dpnn'
local nn = require 'nn'
local nninit = require 'nninit'

cudnn.fastest = true
local backend = cudnn

-- function deepwarp.base.createEncoder()
--   local LEAK = 0.02

--   local net = nn.Sequential()

--   net:add(backend.SpatialConvolution(3, 64, 2, 2, 1, 1):noBias())
--   net:add(backend.SpatialBatchNormalization(64))
--   net:add(nn.LeakyReLU(LEAK))

--   net:add(backend.SpatialConvolution(64, 128, 7, 7, 2, 2):noBias())
--   net:add(backend.SpatialBatchNormalization(128))
--   net:add(nn.LeakyReLU(LEAK))

--   net:add(backend.SpatialConvolution(128, 256, 5, 5, 2, 2):noBias())
--   net:add(backend.SpatialBatchNormalization(256))
--   net:add(nn.LeakyReLU(LEAK))

--   net:add(backend.SpatialConvolution(256, 256, 7, 7, 2, 2):noBias())
--   net:add(backend.SpatialBatchNormalization(256))
--   net:add(nn.LeakyReLU(LEAK))

--   net:add(backend.SpatialConvolution(256, 512, 4, 4, 1, 1):noBias())
--   net:add(backend.SpatialBatchNormalization(512))
--   net:add(nn.LeakyReLU(LEAK))

--   net:add(nn.View(-1):setNumInputDims(3))
--   net:add(nn.Dropout(0.5))

--   -- Initialize network.
--   local function init(module)
--     local typename = torch.type(module)
--     if typename:find('SpatialConvolution') then
--       module:init('weight', nninit.kaiming,
--                   {gain = {'lrelu', leakiness = LEAK}})
--     end
--   end
--   net:applyToModules(init)

--   return net
-- end


function deepwarp.base.createAEncoder(use_conv_drop)
  local LEAK = 0.02

  if use_conv_drop == nil then
    use_conv_drop = true
  end

  local net = nn.Sequential()

  net:add(nn.SpatialDropout(0.05))

  net:add(backend.SpatialConvolution(3, 64, 5, 5, 2, 2, 2, 2))
  net:add(nn.LeakyReLU(LEAK))

  net:add(backend.SpatialConvolution(64, 128, 5, 5, 2, 2, 2, 2):noBias())
  net:add(backend.SpatialBatchNormalization(128))
  net:add(nn.LeakyReLU(LEAK))

  net:add(backend.SpatialConvolution(128, 256, 5, 5, 2, 2, 2, 2):noBias())
  net:add(backend.SpatialBatchNormalization(256))
  net:add(nn.LeakyReLU(LEAK))
  if use_conv_drop then
    net:add(nn.SpatialDropout(0.5))
  end

  net:add(backend.SpatialConvolution(256, 256, 5, 5, 2, 2, 2, 2):noBias())
  net:add(backend.SpatialBatchNormalization(256))
  net:add(nn.LeakyReLU(LEAK))
  if use_conv_drop then
    net:add(nn.SpatialDropout(0.5))
  end

  net:add(backend.SpatialConvolution(256, 512, 4, 4, 1, 1):noBias())
  net:add(backend.SpatialBatchNormalization(512))
  net:add(nn.LeakyReLU(LEAK))

  net:add(nn.View(-1):setNumInputDims(3))
  net:add(nn.Dropout(0.5))

  net:add(nn.Linear(512, 512):noBias())
  net:add(backend.BatchNormalization(512))
  net:add(nn.LeakyReLU(LEAK))
  net:add(nn.Dropout(0.5))

  -- Initialize network.
  local function init(module)
    local typename = torch.type(module)
    if typename:find('SpatialConvolution') then
      module:init('weight', nninit.kaiming,
                  {gain = {'lrelu', leakiness = LEAK}})
    end
  end
  net:applyToModules(init)

  return net
end

function deepwarp.base.getBeheadedNet(net, num_to_drop)
  local beheaded = net:clone(
      'weight', 'bias', 'gradWeight', 'gradBias')
  for _ = 1, num_to_drop do
    beheaded:remove()
  end
  return beheaded
end

function deepwarp.base.createDClassifier(use_full)
  local net
  if use_full then
    net = deepwarp.base.createAEncoder(false)
  else
    net = nn.Sequential()
  end

  net:add(nn.Linear(512, 1))
  local top_linear = net.modules[#net.modules]
  net:add(backend.Sigmoid())

  -- Initialize network.
  top_linear:init('weight', nninit.kaiming, {gain = 'sigmoid'})
            :init('bias', nninit.constant, 0.0)
  
  return net
end

function deepwarp.base.createAClassifier(use_full)
  local net
  if use_full then
    net = deepwarp.base.createAEncoder()
  else
    net = nn.Sequential()
  end

  net:add(nn.Linear(512, 1))
  local top_linear = net.modules[#net.modules]
  net:add(backend.Sigmoid())

  -- Initialize network.
  top_linear:init('weight', nninit.kaiming, {gain = 'sigmoid'})
            :init('bias', nninit.constant, 0.0)
  
  return net
end

function deepwarp.base.attachHead(net, num_outputs, non_linearity)
  num_outputs = num_outputs or 1
  non_linearity = non_linearity or 'sigmoid'

  net:add(nn.Linear(512, num_outputs))
  local top_linear = net.modules[#net.modules]

  if non_linearity == 'sigmoid' then
    net:add(backend.Sigmoid())
  end

  -- Initialize network.
  top_linear:init('weight', nninit.kaiming, {gain = non_linearity})
            :init('bias', nninit.constant, 0.0)
  
  return net
end
