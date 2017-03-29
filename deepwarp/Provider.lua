local npy4th = require 'npy4th'
local Provider = torch.class('deepwarp.Provider')

function Provider:__init(path, mu, disable_random, use_all_attrs)
  self.use_all_attrs = use_all_attrs or false

  self.data = npy4th.loadnpz(path)

  self.data.images = self.data.images:type('torch.FloatTensor')
  self.data.src_labels = self.data.labels:type('torch.FloatTensor')
  self.data.anchors = self.data.anchors:type('torch.FloatTensor')
  if not self.use_all_attrs then
    self.data.src_labels = self.data.src_labels[{ {}, 32 }]
    self.data.src_labels = self.data.src_labels:resize(
        self.data.src_labels:nElement())
  end
  collectgarbage()

  print('Labels range:')
  print(self.data.src_labels:min(), self.data.src_labels:max())
  self.data.images:div(255.0)

  self.mu = mu or self.data.images:mean(1)
  self.gpu_mu = torch.CudaTensor(self.mu:size()):copy(self.mu)
  -- print(self.mu:size())

  self.cur_index = 1
  self.total_indices = self.data.images:size(1)

  if disable_random then
    self.randomize_perm = false
    self.perm = torch.range(1, self.total_indices)
  else
    self.randomize_perm = true
    self.perm = torch.LongTensor()
  end

  self.batch_indices = torch.LongTensor()

  -- Setup buffers.
  self.cpu_buffers = {
    images = torch.FloatTensor(),
    images_m_mu = torch.FloatTensor(),
    anchors = torch.FloatTensor(),
    src_labels = torch.FloatTensor(),
    dst_labels = torch.FloatTensor(),
    deltas = torch.FloatTensor(),
  }

  self.buffers = {
    images = torch.CudaTensor(),
    images_m_mu = torch.CudaTensor(),
    anchors = torch.CudaTensor(),
    anchor_maps = torch.CudaTensor(),
    src_labels = torch.CudaTensor(),
    dst_labels = torch.CudaTensor(),
    all_labels = torch.CudaTensor(),
    domains = torch.CudaTensor(),
    one_m_domains = torch.CudaTensor(),
    deltas = torch.CudaTensor(),
  }
end

function Provider:fillDeltas()
  local deltas = self.cpu_buffers.deltas
  local src_labels = self.cpu_buffers.src_labels
  local dst_labels = self.cpu_buffers.dst_labels
  dst_labels:resizeAs(src_labels):copy(src_labels)
  deltas:resizeAs(src_labels):zero()

  local deltas_roi = deltas
  local src_labels_roi = src_labels
  local dst_labels_roi = dst_labels
  if self.use_all_attrs then
    deltas_roi = deltas[{ {}, 32 }]
    src_labels_roi = src_labels[{ {}, 32 }]
    dst_labels_roi = dst_labels[{ {}, 32 }]
  end

  deltas_roi:fill(1.0)
  self.deltas_mul = self.deltas_mul or torch.FloatTensor(deltas_roi:size())
  self.deltas_mul:copy(src_labels_roi):mul(-2.0):add(1.0)
  deltas_roi:cmul(self.deltas_mul)

  dst_labels_roi:copy(src_labels_roi):add(deltas_roi)

  self.cpu_buffers.deltas = deltas:view(src_labels:size(1), -1)
end

function Provider:createAnchorMaps()
  local NUM_ANCHORS = 5

  local h = self.data.images:size(3)
  local w = self.data.images:size(4)

  if not self.gpu_xs_ys then
    local cpu_xs = torch.range(0, w - 1)
    local cpu_ys = torch.range(0, h - 1)

    self.gpu_xs = torch.CudaTensor()
    self.gpu_ys = torch.CudaTensor()
    self.gpu_xs_ys = torch.CudaTensor()

    self.gpu_xs:resize(cpu_xs:size()):copy(cpu_xs)
    self.gpu_ys:resize(cpu_ys:size()):copy(cpu_ys)

    self.gpu_xs_ys:cat(self.gpu_xs:view(1, 1, w):expand(1, h, w), 
                       self.gpu_ys:view(1, h, 1):expand(1, h, w), 1)
  end

  local batch_size = self.buffers.anchors:size(1)

  self.buffers.anchor_maps:add(
      self.gpu_xs_ys:view(1, 1, 2, h, w)
                    :expand(batch_size, NUM_ANCHORS, 2, h, w), 
      -1,
      self.buffers.anchors:view(batch_size, NUM_ANCHORS, 2, 1, 1)
                          :expand(batch_size, NUM_ANCHORS, 2, h, w))

  self.buffers.anchor_maps[{ {}, {}, 1, {}, {} }]:div(w)
  self.buffers.anchor_maps[{ {}, {}, 2, {}, {} }]:div(h)

  self.buffers.anchor_maps:resize(batch_size, NUM_ANCHORS * 2, h, w)
end

function Provider:nextBatch(batch_size)
  if self.cur_index == 1 and self.randomize_perm then
    self.perm:randperm(self.total_indices)
  end

  self.batch_indices:resize(batch_size):zero()
  for i = 0, batch_size - 1 do
    local j = (self.cur_index - 1 + i) % self.total_indices
    self.batch_indices[i + 1] = self.perm[j + 1] 
  end
  
  -- Gather CPU batch.
  for k, v in pairs(self.cpu_buffers) do
    if self.data[k] then
      local chunk_size = self.data[k]:size()
      chunk_size[1] = batch_size
      v:resize(chunk_size)
      v[{ {1, batch_size} }]:copy(self.data[k]:index(1, self.batch_indices))
    end
  end

  self:fillDeltas()

  -- -- Soften attribute labels.
  -- self.cpu_buffers.src_labels:mul(0.9):add(0.05)

  -- Subtract mean.
  self.cpu_buffers.images_m_mu:resizeAs(self.cpu_buffers.images)
                              :add(self.cpu_buffers.images, -1.0, 
                                   self.mu:expandAs(self.cpu_buffers.images))

  -- Copy data to the GPU.
  for k, v in pairs(self.buffers) do
    if self.cpu_buffers[k] then
      v:resize(self.cpu_buffers[k]:size()):copy(self.cpu_buffers[k])
    end
  end
  self:createAnchorMaps()

  -- Concatenate source and destination labels into a single buffer.
  self.buffers.all_labels:resize(2 * batch_size,
                                 self.buffers.src_labels:size(2))
  self.buffers.all_labels:narrow(1, 1, batch_size)
                         :copy(self.buffers.src_labels)
  self.buffers.all_labels:narrow(1, batch_size + 1, batch_size)
                         :copy(self.buffers.dst_labels)

  -- Set domain labels.
  if self.buffers.domains:nElement() ~= 2 * batch_size then  
    self.buffers.domains:resize(2 * batch_size)
    self.buffers.domains:narrow(1, 1, batch_size):fill(1.0) --:fill(1.0)
    self.buffers.domains:narrow(1, batch_size + 1, batch_size):fill(0.0)
    self.buffers.one_m_domains:resizeAs(self.buffers.domains)
                              :copy(self.buffers.domains)
                              :mul(-1.0)
                              :add(1.0)
  end

  self.cur_index = self.cur_index + batch_size
  
  local epoch_done = false
  if self.cur_index > self.total_indices then
    self.cur_index = 1
    epoch_done = true
  end

  return epoch_done
end

function Provider:getProgress()
  return self.cur_index, self.total_indices
end
