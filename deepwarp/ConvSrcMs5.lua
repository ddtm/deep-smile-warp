-- [SublimeLinter luacheck-globals:+cudnn,deepwarp,nn]

require 'cudnn'
-- require 'dpnn'
require 'nn'
require 'nngraph'
require 'stn'

local nninit = require 'nninit'
local pltx = require 'pl.tablex'

cudnn.fastest = true
local backend = cudnn

local SrcNet, parent = torch.class('deepwarp.base.ConvSrcMs5', 'nn.Container')

function SrcNet:__init(opts)
  -- local SCALE_16_MAPS = {3, 16, 48, 64, 64, 4}
  -- local SCALE_32_MAPS = {3 + 4, 16, 48, 64, 64, 4}
  -- local SCALE_64_MAPS = {3 + 4, 16, 48, 64, 64, 4}

  local SCALE_16_MAPS = {3,     32, 96, 128, 128, 8}
  local SCALE_32_MAPS = {3 + 8, 32, 96, 128, 128, 8}
  local SCALE_64_MAPS = {3 + 8, 32, 96, 128, 128, 2 + 1 + opts.palette_size}

  parent.__init(self)

  local input_m_mu_node = nn.Identity()()
  local delta_node = nn.Identity()()
  local f

  local model_inputs = {input_m_mu_node, delta_node}

  local input_maps_node = input_m_mu_node

  if opts.use_anchors then
    SCALE_16_MAPS[1] = SCALE_16_MAPS[1] + 10
    SCALE_32_MAPS[1] = SCALE_32_MAPS[1] + 10
    SCALE_64_MAPS[1] = SCALE_64_MAPS[1] + 10

    local anchors_node = nn.Identity()()
    table.insert(model_inputs, anchors_node)
    input_maps_node = nn.JoinTable(1, 3)({input_maps_node, anchors_node})
  elseif opts.append_location then
    SCALE_16_MAPS[1] = SCALE_16_MAPS[1] + 2
    SCALE_32_MAPS[1] = SCALE_32_MAPS[1] + 2
    SCALE_64_MAPS[1] = SCALE_64_MAPS[1] + 2

    local zeros = torch.Tensor(2, 64, 64):zero()
    local zeros_node = deepwarp.Constant(zeros, 3)(input_maps_node)
    local location_node = deepwarp.AddLoc()(zeros_node)
    input_maps_node = nn.JoinTable(1, 3)({input_maps_node, location_node})
  end

  --
  -- Calculate source pixels.
  --
  ----------------------------------------------------------------- 16x16 scale
  local scale_16_net = self._createConvModule(
      16, opts.delta_vec, SCALE_16_MAPS)

  f = nn.SpatialAveragePooling(4, 4, 4, 4, 0, 0)(input_maps_node)
  f = backend.Tanh()(scale_16_net({f, delta_node}))
  f = nn.SpatialUpSamplingNearest(2)(f)
  f = nn.SpatialAveragePooling(3, 3, 1, 1, 1, 1)(f)

  local scale_16_net_node = f

  ----------------------------------------------------------------- 32x32 scale
  local scale_32_net = self._createConvModule(
      32, opts.delta_vec, SCALE_32_MAPS)

  f = nn.SpatialAveragePooling(2, 2, 2, 2, 0, 0)(input_maps_node)
  f = nn.JoinTable(1, 3)({f, scale_16_net_node})
  f = backend.Tanh()(scale_32_net({f, delta_node}))
  f = nn.SpatialUpSamplingNearest(2)(f)
  f = nn.SpatialAveragePooling(3, 3, 1, 1, 1, 1)(f)

  local scale_32_net_node = f

  ----------------------------------------------------------------- 64x64 scale
  local scale_64_net = self._createConvModule(
      64, opts.delta_vec, SCALE_64_MAPS)

  f = input_maps_node
  f = nn.JoinTable(1, 3)({f, scale_32_net_node})
  f = scale_64_net({f, delta_node})

  -- Split output into two parts: the flow and LCM weights.
  local f1 = nn.Narrow(2, 1, 2)(f)
  local f2 = nn.Narrow(2, 3, -1)(f)

  local flow_node = backend.Tanh()(f1)

  local lcm_bias = torch.Tensor(1 + opts.palette_size):zero()
  lcm_bias[1] = 0.0
  lcm_bias = lcm_bias:view(1 + opts.palette_size, 1, 1)
                     :expand(1 + opts.palette_size, 64, 64)
                     :contiguous()
  local lcm_bias_node = deepwarp.Constant(lcm_bias, 3)(f2)

  f2 = nn.CAddTable(true)({f2, lcm_bias_node})
  f2 = backend.SpatialSoftMax()(f2)

  if opts.flow_tv > 0 then
    flow_node = deepwarp.TVLoss(opts.flow_tv)(flow_node)
  end
  if opts.flow_l1 > 0 then
    flow_node = nn.L1Penalty(opts.flow_l1)(flow_node)
  end

  local f21 = nn.Narrow(2, 1, 1)(f2)
  local f22 = nn.Narrow(2, 2, -1)(f2)
  if opts.lcm_tv > 0 then
    f22 = deepwarp.TVLoss(opts.lcm_tv)(f22)
  end
  if opts.lcm_l1 > 0 then
    f22 = nn.L1Penalty(opts.lcm_l1)(f22)
  end
  local mixture_weights_node = nn.JoinTable(1, 3)({f21, f22})

  flow_node = nn.MulConstant(0.1)(flow_node)

  local bhwd_disp_node = nn.Transpose({3, 4}, {2, 4})(
      deepwarp.AddLoc()(flow_node))

  -------------------------------------------------------- Compose full network
  self.src_net = nn.gModule(model_inputs,
                            {bhwd_disp_node, mixture_weights_node})

  self.modules = {self.src_net}
  self.x_m_mu = torch.Tensor()
end

function SrcNet:disablePenalties()
  for _, v in pairs(self:findModules('deepwarp.TVLoss')) do
    print(v)
    v.strength = 0.0
  end
  for _, v in pairs(self:findModules('nn.L1Penalty')) do
    print(v)
    v.l1weight = 0.0
  end
end

function SrcNet._createConvModule(image_size, delta_size, num_maps)
  local input_node = nn.Identity()()
  local delta_node = nn.Identity()()

  local function depth_concat(x, y, height, width)
    y = nn.Transpose({1, 3}, {2, 4})(
        nn.View(height, width, -1, delta_size):setNumInputDims(3)(
            nn.Contiguous()(nn.Replicate(height * width)(y))))
    return nn.JoinTable(1, 3)({x, y})
  end

  local function conv_block(x, height, num_in, num_out, ...)
    num_in = num_in + delta_size
    local conv = backend.SpatialConvolution(num_in, num_out, ...):noBias()
    local bn = backend.SpatialBatchNormalization(num_out)
    local relu = backend.ReLU()
    return relu(bn(conv(depth_concat(x, delta_node, height, height))))
  end

  local f = input_node
  f = conv_block(f, image_size, num_maps[1], num_maps[2], 5, 5, 1, 1, 2, 2)
  f = conv_block(f, image_size, num_maps[2], num_maps[3], 3, 3, 1, 1, 1, 1)
  f = conv_block(f, image_size, num_maps[3], num_maps[4], 3, 3, 1, 1, 1, 1)
  f = conv_block(f, image_size, num_maps[4], num_maps[5], 1, 1, 1, 1)
  local top_conv = backend.SpatialConvolution(
      num_maps[5], num_maps[6], 1, 1, 1, 1)
  f = top_conv(f)

  local net = nn.gModule({input_node, delta_node}, {f})

  local function init(module)
    local typename = torch.type(module)
    if typename:find('SpatialConvolution') then
      module:init('weight', nninit.kaiming, {gain = 'relu'})
      if module.bias ~= nil then
        module:init('bias', nninit.constant, 0.0)
      end
    end
  end
  net:applyToModules(init)

  top_conv:init('weight', nninit.normal, 0, 0.01)
          :init('bias', nninit.constant, 0.01)

  return net
end

function SrcNet:updateOutput(input)
  -- Replace mean with x - mean in the input.
  local x, mu = input[1], input[2]
  self.x_m_mu:resizeAs(x):add(x, -1.0, mu:expandAs(x))
  self.input = pltx.copy(input)
  self.input[1] = self.x_m_mu
  table.remove(self.input, 2)

  self.output = self.src_net:updateOutput(self.input)
  return self.output
end

function SrcNet:updateGradInput(_, gradOutput)
  self.gradInput = self.src_net:updateGradInput(self.input, gradOutput)

  -- Insert dummy gradient for the mean.
  table.insert(self.gradInput, 2, nil)
  
  return self.gradInput
end

function SrcNet:accGradParameters(_, gradOutput, scale)
  self.src_net:accGradParameters(self.input, gradOutput, scale)
end
