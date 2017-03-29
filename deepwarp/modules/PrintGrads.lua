local PrintGrads, parent = torch.class('deepwarp.modules.PrintGrads', 'nn.Module')

function PrintGrads:__init(prefix)
   parent.__init(self)
   self.prefix = prefix or ''
end

function PrintGrads:updateOutput(input)
   self.output = input
   return self.output
end

function PrintGrads:updateGradInput(input, gradOutput)
   print(self.prefix .. ' gradient norms:')
   if torch.type(gradOutput) == 'table' then
      for i = 1, #gradOutput do
         if torch.type(gradOutput) == 'nil' then
            print(('  %2d: nil'):format(i))
         else
            print(('  %2d: %.8f'):format(i, gradOutput[i]:norm()))
         end
      end
   elseif torch.type(gradOutput) == 'nil' then
      print('  1: nil')
   else
      print(('  1: %.8f'):format(gradOutput:norm()))
   end
   self.gradInput = gradOutput
   return self.gradInput
end
