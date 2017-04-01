-- [SublimeLinter luacheck-globals:+lapp,nn,paths,xlua]

package.path = package.path .. ";./?/init.lua"

local deepwarp = require 'deepwarp'
local nn = require 'nn'
local optim = require 'optim'
local utils = require 'utils'
local xlua = require 'xlua'

local c = require 'trepl.colorize'

require 'cunn'

utils.dispImages = utils.dispImagesTGAN
utils.dispPlot = utils.dispPlotTGAN

local opts = lapp[[
  -d,--data_path       (default ./)            Data path
  -s,--snapshot_path   (default ./snapshots)   Snapshots path
  -f,--snapshot_freq   (default 50)            Snapshots frequency
  -t,--test_freq       (default 20)            Test frequency
  -l,--learning_rate   (default 1e-4)          Learning rate
  -b,--batch_size      (default 128)           Batch size
  --beta1              (default 0.5)           Adam: beta_1
  --delta_vec          (default 4)             Dimensionality of delta vec
  --palette_size       (default 1)             Number of colors in palette
  --append_location                            Append location maps to the input
  --use_anchors                                Use anchors
  --use_all_attrs                              Use all binary attributes
  --main_attr_weight   (default 1.0)           Weight of the main attribute
  --transformer        (default Transformer5) 
  --noblur_half_scale                          Disable blurring of half scale shifts
  --gan_loss_weight    (default 1.0)           GAN loss weight
  --attr_loss_weight   (default 1.0)           Attribure loss weight
  --flow_tv            (default 0)             Strength of TV denoising for flow
  --flow_l1            (default 0)             Strength of L1 penalty for flow
  --lcm_tv             (default 0)             Strength of TV denoising for LCM
  --lcm_l1             (default 0)             Strength of L1 penalty for LCM
  --window_idx         (default 10)            Window index
  --window_postfix     (default '')            Window postfix
]]

opts.blur_half_scale = not opts.noblur_half_scale
opts.noblur_half_scale = nil

print(opts)

local plot_data = {}

local function train(model, provider, criterion, optimizer)
  model:training()

  local input = {provider.buffers.images, provider.gpu_mu, 
                 provider.buffers.deltas}
  if opts.use_anchors then
    table.insert(input, provider.buffers.anchor_maps)
  end
  local target = {provider.buffers.domains, provider.buffers.one_m_domains,
                  provider.buffers.all_labels}
  local grad_divide_by = {2 * opts.batch_size, 2 * opts.batch_size, 
                          opts.batch_size}

  local losses = {}
  for i = 1, #criterion.criterions do
    losses[i] = {}
  end

  local epoch_done = false
  local iter = 0

  while not epoch_done do
    local cur, total = provider:getProgress()
    xlua.progress(cur, total)

    epoch_done = provider:nextBatch(opts.batch_size)

    local feval = function(x)
      if x ~= model.flat_params then 
        model.flat_params:copy(x) 
      end
      model.flat_grad_params:zero()
      
      local output = model:forward(input)
      local f = criterion:forward(output, target) / (2 * opts.batch_size)
      local df_do = criterion:backward(output, target)
      for i = 1, #df_do do
        df_do[i]:div(grad_divide_by[i])
      end
      model:backward(input, df_do)

      return f, model.flat_grad_params
    end

    optimizer.fun(feval, model.flat_params, optimizer.state)

    for i = 1, #criterion.criterions do
      losses[i][#losses[i] + 1] = 
          criterion.criterions[i].output / (2 * opts.batch_size)
    end
    iter = iter + 1

    if iter % 10 == 0 then
      utils.dispImages(model, provider, opts.window_idx + 10,
                       'Train' .. opts.window_postfix)
      local plot_entry = {#plot_data}
      for i = 1, #criterion.criterions do
        table.insert(plot_entry, losses[i][#losses[i]])
      end
      table.insert(plot_data, plot_entry)
      utils.dispPlot(plot_data, opts.window_idx + 20,
                     'Train' .. opts.window_postfix)
    end
  end
  local _, total = provider:getProgress()
  xlua.progress(total, total)

  local mean_losses = {}
  for i = 1, #criterion.criterions do
    mean_losses[i] = torch.mean(torch.FloatTensor(losses[i]))
  end
  return mean_losses
end

local function test(model, provider, criterion)
  model:evaluate()

  local input = {provider.buffers.images, provider.gpu_mu, 
                provider.buffers.deltas}
  if opts.use_anchors then
    table.insert(input, provider.buffers.anchor_maps)
  end
  local target = {provider.buffers.domains, provider.buffers.one_m_domains, 
                  provider.buffers.all_labels}

  local losses = {}
  for i = 1, #criterion.criterions do
    losses[i] = {}
  end

  local epoch_done = false
  
  while not epoch_done do
    local cur, total = provider:getProgress()
    xlua.progress(cur, total)

    epoch_done = provider:nextBatch(opts.batch_size)
      
    local output = model:forward(input)
    criterion:forward(output, target)

    for i = 1, #criterion.criterions do
      losses[i][#losses[i] + 1] = 
          criterion.criterions[i].output / (2 * opts.batch_size)
    end
  end
  local _, total = provider:getProgress()
  xlua.progress(total, total)

  local mean_losses = {}
  for i = 1, #criterion.criterions do
    mean_losses[i] = torch.mean(torch.FloatTensor(losses[i]))
  end
  return mean_losses
end

---------------------------------------------------------- Create training task
-- Handle attributes settings.
local num_attrs = 1
local attr_weights = nil
if opts.use_all_attrs then
  num_attrs = 40
  if opts.main_attr_weight > 1.0 then
    attr_weights = torch.Tensor(num_attrs):fill(1.0)
    attr_weights[32] = opts.main_attr_weight
    attr_weights:div(attr_weights:sum())
  end
end

-- Setup providers.
local train_data_path = paths.concat(opts.data_path, 'train.npz')
local test_data_path = paths.concat(opts.data_path, 'test.npz')
-- local train_data_path = paths.concat(opts.data_path, 'train_small.npz')
-- local test_data_path = paths.concat(opts.data_path, 'train_small.npz')

local train_provider = deepwarp.Provider(
    train_data_path, nil, false, opts.use_all_attrs)
local test_provider = deepwarp.Provider(
    test_data_path, train_provider.mu, false, opts.use_all_attrs)

-- Create models.
local transformer = deepwarp[opts.transformer](opts)
local d_encoder = deepwarp.base.createAEncoder()
local a_encoder = d_encoder:clone('weight', 'bias', 'gradWeight', 'gradBias')
local d_classifier = deepwarp.base.attachHead(d_encoder)
local a_classifier = deepwarp.base.attachHead(a_encoder, num_attrs)

local tgan = deepwarp.TGAN(transformer, d_classifier, a_classifier)
tgan:cuda()
tgan:flattenParams()

-- Setup training criteria.
local d_criterion = nn.BCECriterion()
d_criterion.sizeAverage = false
local g_criterion = nn.BCECriterion()
g_criterion.sizeAverage = false
local a_criterion = nn.BCECriterion(attr_weights)
a_criterion.sizeAverage = false

local train_criterion =
    nn.ParallelCriterion():add(d_criterion, opts.gan_loss_weight)
                          :add(g_criterion, opts.gan_loss_weight)
                          :add(a_criterion, opts.attr_loss_weight)
train_criterion:cuda()

d_criterion = nn.BCECriterion()
d_criterion.sizeAverage = false
g_criterion = nn.BCECriterion()
g_criterion.sizeAverage = false
a_criterion = nn.BCECriterion(attr_weights)
a_criterion.sizeAverage = false

local test_criterion =
    nn.ParallelCriterion():add(d_criterion, opts.gan_loss_weight)
                          :add(g_criterion, opts.gan_loss_weight)
                          :add(a_criterion, opts.attr_loss_weight)
test_criterion:cuda()

-- Setup optimizer.
local optimizer = {
  fun = optim.adam,
  state = {
    learningRate = opts.learning_rate,
    weightDecay = 0.0,
    beta1 = opts.beta1
  }
}

------------------------------------------------------------------ Run training
local snapshot_template = paths.concat(opts.snapshot_path, '%d.t7')
local start_iter = 1
local end_iter = 2400

tgan:saveParams(snapshot_template:format(start_iter - 1))

if opts.window_postfix ~= '' then
  opts.window_postfix = ' (' .. opts.window_postfix .. ')'
end

for epoch = start_iter, end_iter do
  print(c.blue '==>' .. ' online epoch # ' .. epoch .. ' [batch size = ' .. 
        opts.batch_size .. ']')
  local tic = torch.tic()

  local mean_losses = train(tgan, train_provider, train_criterion, optimizer)
  print(('Mean train D loss: ' .. c.cyan '%.8f\n' ..
         'Mean train G loss: ' .. c.cyan '%.8f\n' ..
         'Mean train A loss: ' .. c.cyan '%.8f\n' .. 
         'Time: %.2f s'):format(
        mean_losses[1], mean_losses[2], mean_losses[3], torch.toc(tic)))

  utils.dispImages(tgan, train_provider, opts.window_idx + 10,
                   'Train' .. opts.window_postfix)

  if epoch % opts.test_freq == 0 then
    mean_losses = test(tgan, test_provider, test_criterion)
    print(('Mean test D loss: ' .. c.cyan '%.8f\n' ..
           'Mean test G loss: ' .. c.cyan '%.8f\n' ..
           'Mean test A loss: ' .. c.cyan '%.8f'):format(
          mean_losses[1], mean_losses[2], mean_losses[3]))
    
    utils.dispImages(tgan, test_provider, opts.window_idx,
                     'Test' .. opts.window_postfix)
  end

  if epoch % opts.snapshot_freq == 0 then
    tgan:saveParams(snapshot_template:format(epoch))
  end
end
  
