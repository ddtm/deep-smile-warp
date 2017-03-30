-- [SublimeLinter luacheck-globals:+deepwarp]

deepwarp = {}
include 'AddLoc.lua'
include 'AnchorsToMaps.lua'
include 'Constant.lua'
include 'TrainableTensor.lua'
include 'TVLoss.lua'
include 'Provider.lua'
include 'Transformer5.lua'
include 'TGAN.lua'

deepwarp.modules = {}
include 'modules/PrintGrads.lua'
include 'modules/ScaleGrads.lua'
include 'modules/SpatialReplication.lua'
include 'modules/SplitBatch.lua'

deepwarp.base = {}
include 'ConvSrcMs5.lua'
include 'base/init.lua'

return deepwarp
