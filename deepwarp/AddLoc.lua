require 'nn'

local AddLoc, parent = torch.class('deepwarp.AddLoc', 'nn.Module')

function AddLoc:__init()
  parent.__init(self)

  self.locations = torch.Tensor()
  self.output = torch.Tensor()
end

function AddLoc:updateOutput(input)
  local h = input:size(3)
  local w = input:size(4)

  local locations = self.locations

  if locations:nElement() == 0 or 
     h ~= locations:size(2) or 
     w ~= locations:size(3) then 
    locations:resize(2, h, w)
    locations[{ 1, {}, {} }] = torch.linspace(-1.0, 1.0, h):view(h, 1)
                                                           :expand(h, w)
    locations[{ 2, {}, {} }] = torch.linspace(-1.0, 1.0, w):view(1, w)
                                                           :expand(h, w)
  end

  locations = locations:view(1, input:size(2), input:size(3), input:size(4))
  locations = locations:expandAs(input)
  self.output:add(input, locations)
  return self.output
end

function AddLoc:updateGradInput(_, gradOutput)
  self.gradInput = gradOutput
  return self.gradInput
end

return AddLoc
