require 'nn'

local AnchorsToMaps, parent = torch.class('deepwarp.AnchorsToMaps', 'nn.Module')

function AnchorsToMaps:__init(height, width)
  parent.__init(self)

  self.height = height
  self.width = width

  self.xs = torch.Tensor()
  self.ys = torch.Tensor()
  self.xs_ys = torch.Tensor()

  self.locations = torch.Tensor()
  self.output = torch.Tensor()
end

function AnchorsToMaps:updateOutput(input)
  local h = self.height
  local w = self.width

  if not self.xs_ys:nElement() == 0 then
    local cpu_xs = torch.range(0, w - 1)
    local cpu_ys = torch.range(0, h - 1)

    self.xs:resize(cpu_xs:size()):copy(cpu_xs)
    self.ys:resize(cpu_ys:size()):copy(cpu_ys)

    self.xs_ys:cat(self.xs:view(1, 1, w):expand(1, h, w), 
                   self.ys:view(1, h, 1):expand(1, h, w), 1)
  end

  local batch_size = input:size(1)
  local num_anchors = input:size(2)

  self.output:add(
      self.xs_ys:view(1, 1, 2, h, w)
                :expand(batch_size, num_anchors, 2, h, w), 
      -1,
      input:view(batch_size, num_anchors, 2, 1, 1)
           :expand(batch_size, num_anchors, 2, h, w))

  self.output[{ {}, {}, 1, {}, {} }]:div(w)
  self.output[{ {}, {}, 2, {}, {} }]:div(h)

  self.output:resize(batch_size, num_anchors * 2, h, w)

  return self.output
end

function AnchorsToMaps:updateGradInput(_, _)
  self.gradInput = nil
  return self.gradInput
end

return AnchorsToMaps
