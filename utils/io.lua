local nn = require 'nn'
local pltx = require 'pl.tablex'

nn.Container.flattenParams = function(self)
  self.flat_params, self.flat_grad_params = self:getParameters()
end

nn.Container.saveParams = function(self, filename)
  local bn_nodes = {}
  for _, v in pairs(self:findModules('cudnn.SpatialBatchNormalization')) do
    table.insert(bn_nodes, v)
  end
  for _, v in pairs(self:findModules('cudnn.BatchNormalization')) do
    table.insert(bn_nodes, v)
  end
  local bn_estimates = {}
  for i = 1, #bn_nodes do
    bn_estimates[#bn_estimates + 1] = {
      running_mean = bn_nodes[i].running_mean,
      running_var = bn_nodes[i].running_var
    }
  end
  torch.save(filename, {flat_params = self.flat_params, 
                        bn_estimates = bn_estimates})
end

nn.Container.loadParams = function(self, filename)
  local params = torch.load(filename)
  self.flat_params:copy(params.flat_params)
  local bn_nodes = {}
  for _, v in pairs(self:findModules('cudnn.SpatialBatchNormalization')) do
    table.insert(bn_nodes, v)
  end
  for _, v in pairs(self:findModules('cudnn.BatchNormalization')) do
    table.insert(bn_nodes, v)
  end
  for i = 1, #bn_nodes do
    bn_nodes[i].running_mean = params.bn_estimates[i].running_mean
    bn_nodes[i].running_var = params.bn_estimates[i].running_var
  end
end

function utils.save(to_save, snapshot_path, iter)
  local snapshot_template = paths.concat(snapshot_path, '%d.t7')
  local optimizer_snapshot_template = paths.concat(snapshot_path, '%d_opt.t7')
  to_save.model:saveParams(snapshot_template:format(iter))
  if to_save.optimizer then
    torch.save(optimizer_snapshot_template:format(iter), to_save.optimizer)
  end
end

function utils.load(to_load, snapshot_path, iter)
  local snapshot_template = paths.concat(snapshot_path, '%d.t7')
  local optimizer_snapshot_template = paths.concat(snapshot_path, '%d_opt.t7')
  to_load.model:loadParams(snapshot_template:format(iter))
  pltx.update(
      to_load.optimizer, torch.load(optimizer_snapshot_template:format(iter)))
end
