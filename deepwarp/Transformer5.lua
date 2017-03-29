-- [SublimeLinter luacheck-globals:+cudnn,deepwarp,nn]

require 'cudnn'
require 'nn'
require 'nngraph'
require 'stn'

local nninit = require 'nninit'
local nnq = require 'nnquery'

cudnn.fastest = true
local backend = cudnn

local Transformer, parent = torch.class('deepwarp.Transformer5', 'nn.Container')

function Transformer:__init(opts)
  parent.__init(self)
  
  self.batch_size = opts.batch_size

  if opts.delta_vec == nil then
    opts.delta_vec = 16
  end

  local input_node = nn.Identity()()
  local mu_node = nn.Identity()()
  local deltas_node = nn.Identity()()
  local model_inputs = {input_node, mu_node, deltas_node}

  ---------------------------- Transform input angle into hidden representation
  local num_attrs = 1
  if opts.use_all_attrs then
    num_attrs = 40
  end
  local delta_net = self.createDeltaNet(num_attrs, opts.delta_vec)
  local delta_net_node = delta_net(deltas_node)

  ----------------------------------------------------- Calculate source pixels
  local src_net_input = {input_node, mu_node, delta_net_node}
  if opts.use_anchors then
    local anchors_node = nn.Identity()()
    table.insert(model_inputs, anchors_node)
    table.insert(src_net_input, anchors_node)
  end

  self.src_net = deepwarp.base.ConvSrcMs5(opts)
  local bhwd_src_node, mixture_weights_node =
      self.src_net(src_net_input):split(2)

  -- Wrap src_net output nodes in identity layers so that we can extract
  -- output tensors later.
  bhwd_src_node =
      nn.Identity()(bhwd_src_node):annotate{name = 'bhwd_src'}
  mixture_weights_node =
      nn.Identity()(mixture_weights_node):annotate{name = 'mixture_weights'}

  ------------------------------------------------------------- Transform input
  local bhwd_input_node = nn.Transpose({3, 4}, {2, 4})(input_node)
  local bhwd_output_node = nn.BilinearSamplerBHWD()({bhwd_input_node, 
                                                     bhwd_src_node})
  local output_node = nn.Transpose({2, 4}, {3, 4})(bhwd_output_node)

  -------------------------------- Post-process output using correction palette
  -- output_node = deepwarp.PrintGrads('before_lcm')(output_node)
  output_node = self:_applyLCM(
      output_node, mixture_weights_node, opts.palette_size)
  -- output_node = deepwarp.PrintGrads('after_lcm')(output_node)

  ---------------------------------------------------------------- Create model
  self.model = nn.gModule(model_inputs, {output_node})

  local model = nnq(self.model)
  self.bhwd_src_module =
      model:descendants()
           :attr{name = 'bhwd_src'}
           :only():module()
  self.mixture_weights_module =
      model:descendants()
           :attr{name = 'mixture_weights'}
           :only():module()

  -- Pass-through parameters of the embedded network
  -- (otherwise no updates will happen).
  self.modules = {self.model}
end

function Transformer:_applyLCM(output_node, mixture_weights_node, palette_size)
  local palette_module = deepwarp.TrainableTensor(1.0, palette_size, 3)
  local palette_node = backend.Sigmoid()(palette_module(output_node))

  -- Mix palette and output using LCM wieghts.
  output_node = nn.View(1, 3, 64, 64)(output_node)
  palette_node = nn.View(palette_size, 3, 64, 64)(
      nn.Transpose({1, 3})(
          nn.Reshape(64, 64, self.batch_size * palette_size * 3)(
              nn.Replicate(64 * 64)(palette_node))))
  mixture_weights_node = 
      nn.Transpose({1, 3}, {1, 2})(
          nn.Reshape(3, self.batch_size, 1 + palette_size, 64, 64)(
              nn.Replicate(3)(mixture_weights_node)))

  local output_and_palette_node = nn.JoinTable(1, 4)(
      {output_node, palette_node})

  local weighted_output_and_palette_node = nn.CMulTable()(
      {output_and_palette_node, mixture_weights_node})

  output_node = nn.Sum(1, 4)(weighted_output_and_palette_node)

  return output_node
end

function Transformer:updateOutput(input)
  self.output = self.model:updateOutput(input)
  return self.output
end

function Transformer:updateGradInput(input, gradOutput)
  self.gradInput = self.model:updateGradInput(input, gradOutput)
  return self.gradInput
end

function Transformer:accGradParameters(input, gradOutput, scale)
  self.model:accGradParameters(input, gradOutput, scale)
end

function Transformer.createDeltaNet(num_attrs, num_hidden)
  num_attrs = num_attrs or 1
  num_hidden = num_hidden or 16

  local trainable_modules = {}

  local net = nn.Sequential()

  local m = nn.Linear(num_attrs, num_hidden)
  trainable_modules.fc_1 = m
  net:add(m)
  net:add(backend.ReLU())

  m = nn.Linear(num_hidden, num_hidden)
  trainable_modules.fc_2 = m
  net:add(m)
  net:add(backend.ReLU())

  -- Initialize network.
  for _, v in pairs(trainable_modules) do
    v:init('weight', nninit.kaiming, {gain = 'relu'})
     :init('bias', nninit.constant, 0.0)
  end

  return net
end
