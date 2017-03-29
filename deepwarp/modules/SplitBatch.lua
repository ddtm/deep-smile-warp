require 'nn'

local SplitBatch, parent = torch.class('deepwarp.modules.SplitBatch', 'nn.Module')


function SplitBatch:__init(num_chunks)
  parent.__init(self)
  self.num_chunks = num_chunks or 2
  self.split_table = nn.SplitTable(1)
end


local function reshapeBatchDim(size, num_chunks)
  local new_size = torch.LongStorage(#size + 1):fill(1)
  for d = 1, #size do
    new_size[d + 1] = size[d]
  end
  new_size[1] = num_chunks
  new_size[2] = new_size[2] / num_chunks
  return new_size
end


function SplitBatch:updateOutput(input)
  self.new_size =
      self.new_size or reshapeBatchDim(input:size(), self.num_chunks)

  self.input_view = input:view(self.new_size)
  self.output = self.split_table:updateOutput(self.input_view)

  return self.output
end


function SplitBatch:updateGradInput(input, gradOutput)
  self.gradInput =
      self.split_table:updateGradInput(self.input_view, gradOutput)
  self.gradInput = self.gradInput:viewAs(input)
  return self.gradInput
end
