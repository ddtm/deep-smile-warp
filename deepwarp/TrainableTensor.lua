local TrainableTensor, parent = torch.class("deepwarp.TrainableTensor", "nn.Module")

function TrainableTensor:__init(stdv, ...)
  parent.__init(self)
  self.weight = torch.Tensor(...)
  self.gradWeight = torch.Tensor(self.weight:size())
  self.gradBuffer = torch.Tensor(self.weight:size())
  self:reset(stdv)
end

function TrainableTensor:reset(stdv)
  stdv = stdv * math.sqrt(3)
  self.weight:uniform(-stdv, stdv)
  return self
end

function TrainableTensor:updateOutput(input)
  local wsize = self.weight:size():totable()
  self.output:resize(input:size(1), table.unpack(wsize))
  local weight = self.weight:view(1, table.unpack(wsize))
  self.output:copy(weight:expand(self.output:size()))
  return self.output
end

function TrainableTensor:updateGradInput(input, _)
  self.gradInput:resizeAs(input):zero()
  return self.gradInput
end

function TrainableTensor:accGradParameters(_, gradOutput, scale)
  self.gradBuffer:sum(gradOutput, 1)
  self.gradWeight:add(scale, self.gradBuffer)
end
