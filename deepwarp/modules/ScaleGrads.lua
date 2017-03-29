local ScaleGrads, parent = torch.class('deepwarp.modules.ScaleGrads', 'nn.Module')

function ScaleGrads:__init(scale)
   parent.__init(self)
   self.scale = scale
   self.gradInput = torch.Tensor()
end

function ScaleGrads:updateOutput(input)
   self.output = input
   return self.output
end

function ScaleGrads:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput):mul(self.scale)
   return self.gradInput
end
