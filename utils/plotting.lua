local deepwarp = require 'deepwarp'
local disp = require 'display'
local image = require 'image'
local pltx = require 'pl.tablex'

disp.configure({hostname='leto16', port=5008})

function utils.dispImagesTGAN(model, provider, win, title)
  local num_to_disp = 20
  local batch_size = provider.batch_indices:nElement()

  local attr_col = 1
  if provider.use_all_attrs then
    attr_col = 32
  end

  local ins = provider.cpu_buffers.images[{ {1, num_to_disp}, {}, {}, {} }]
  local outs = model.transformer.output[{ {1, num_to_disp}, {}, {}, {} }]:float()
  local d_outs_x = model.d_classifier.output[{ {1, num_to_disp}, 1 }]:float()
  local a_outs_x = model.a_classifier.output[{ {1, num_to_disp}, attr_col }]:float()
  local d_outs_x_hat = model.d_classifier.output[{ {batch_size + 1, batch_size + num_to_disp}, 1 }]:float()
  local a_outs_x_hat = model.a_classifier.output[{ {batch_size + 1, batch_size + num_to_disp}, attr_col }]:float()

  local src_labels = provider.cpu_buffers.src_labels[{ {1, num_to_disp}, attr_col }]
  local deltas = provider.cpu_buffers.deltas[{ {1, num_to_disp}, attr_col }]
  local infos = torch.FloatTensor(num_to_disp, 3, 64, 64):zero()

  -- Draw info.
  for i = 1, num_to_disp do
    image.drawText(infos[i], 
        ("%+6.4f\n%+6.4f\n%+6.4f\n%+6.4f\n%+6.4f\n%+6.4f"):format(
            src_labels[i], deltas[i], 
            d_outs_x[i], a_outs_x[i],
            d_outs_x_hat[i], a_outs_x_hat[i]),
        0, 0, {inplace = true})
  end

  local to_disp = torch.cat(ins, outs, 4)
  to_disp = torch.cat(to_disp, infos, 4)

  disp.image(to_disp, {win = win, title = title, nperrow = 1})
end

function utils.dispPlotTGAN(plot_data, win, title)
  local config = {
    title = title,
    labels = {"Iter", "D Loss", "G Loss", "A Loss"},
    win = win,
  }

  disp.plot(plot_data, config)
end
