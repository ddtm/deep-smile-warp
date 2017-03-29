local SpatialReplication, parent = torch.class('deepwarp.modules.SpatialReplication', 'nn.Module')


function SpatialReplication:__init(height, width)
  parent.__init(self)
  self.height = height
  self.width = width
  self.gradInput = torch.Tensor()
end


function SpatialReplication:updateOutput(input)
  local batch_size = input:size(1)
  local dim = input:size(2)
  self.output = input:view(batch_size, dim, 1, 1)
                     :expand(batch_size, dim, self.height, self.width)
  return self.output
end


function SpatialReplication:updateGradInput(input, gradOutput)
  local batch_size = input:size(1)
  local dim = input:size(2)
  gradOutput = gradOutput:view(batch_size, -1, dim)
  self.gradInput:resizeAs(input):sum(gradOutput, 2) 
  return self.gradInput
end
