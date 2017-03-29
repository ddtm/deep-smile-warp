-- [SublimeLinter luacheck-globals:+deepwarp,nn]

require 'nn'

local TGAN, parent = torch.class('deepwarp.TGAN', 'nn.Container')

function TGAN:__init(transformer, d_classifier, a_classifier)
  parent.__init(self)
  self.gradInput = {torch.Tensor(), torch.Tensor()}
  self.x_x_hat_grad = torch.Tensor()
  self.a_output_grad = torch.Tensor()

  self.transformer = transformer
  self.d_classifier = d_classifier
  self.a_classifier = a_classifier
  self.concat = nn.JoinTable(1, 4)

  self.modules = {
    self.transformer,
    self.concat,
    self.d_classifier,
    self.a_classifier
  }

  self.update_d = true
  self.update_a = true
end

function TGAN:updateOutput(input)
  self.input = input
  self.x = self.input[1]
  self.mu = self.input[2]
  self.delta = self.input[3]
             
  self.x_hat = self.transformer:updateOutput(self.input)
  self.x_x_hat = self.concat:updateOutput({self.x, self.x_hat})
  local d_output = self.d_classifier:updateOutput(self.x_x_hat)
  local a_output = self.a_classifier:updateOutput(self.x_x_hat)
  self.output = {d_output, d_output, a_output}
  
  return self.output
end

function TGAN:updateGradInput(_, gradOutput)
  self:_updateXHatGrad(gradOutput)

  -- Backprop through the transformer.
  self.transformer:updateGradInput(self.input, 
                                   self.x_hat_grad)

  return self.gradInput
end

function TGAN:_updateXHatGrad(gradOutput)
  local d_output_grad, g_output_grad, a_output_grad = unpack(gradOutput)

  -- Backprop generator gradients through the discriminator.
  local x_x_hat_g_grad = self.d_classifier:updateGradInput(
      self.x_x_hat, g_output_grad)
  self.x_x_hat_grad:resizeAs(x_x_hat_g_grad):copy(x_x_hat_g_grad)

  -- Backprop discriminator gradients through the discriminator.
  self.d_classifier:updateGradInput(self.x_x_hat, d_output_grad)

  -- Backprop x_hat through the attribute branch.
  local batch_size = self.x:size(1)
  self.a_output_grad:resizeAs(a_output_grad):copy(a_output_grad)
  self.a_output_grad:narrow(1, 1, batch_size):zero()
  local x_x_hat_a_grad = self.a_classifier:updateGradInput(
      self.x_x_hat, self.a_output_grad)
  self.x_x_hat_a_grad = x_x_hat_a_grad
  self.x_x_hat_grad:add(x_x_hat_a_grad)

  -- Backprop x through the attribute branch.
  if self.update_a then
    self.a_output_grad:copy(a_output_grad)
    self.a_output_grad:narrow(1, batch_size + 1, batch_size):zero()
    self.a_classifier:updateGradInput(
        self.x_x_hat, self.a_output_grad)
  end

  local _, x_hat_grad = unpack(
      self.concat:updateGradInput({self.x, self.x_hat}, self.x_x_hat_grad))
  self.x_hat_grad = x_hat_grad
end

function TGAN:accGradParameters(_, gradOutput, scale)
  scale = scale or 1
  local d_output_grad, _, a_output_grad = unpack(gradOutput)

  if self.update_d then
    self.d_classifier:accGradParameters(self.x_x_hat, d_output_grad, scale)
  end
  if self.update_a then
    local batch_size = self.x:size(1)
    self.a_output_grad:resizeAs(a_output_grad):copy(a_output_grad)
    self.a_output_grad:narrow(1, batch_size + 1, batch_size):zero()
    self.a_classifier:accGradParameters(
        self.x_x_hat, self.a_output_grad, scale)
  end

  -- Update the transformer param grads.
  self.transformer:accGradParameters(self.input, self.x_hat_grad, scale)
end
