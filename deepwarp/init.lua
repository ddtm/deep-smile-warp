-- [SublimeLinter luacheck-globals:+deepwarp]

deepwarp = {}
include 'AddLoc.lua'
include 'AnchorsToMaps.lua'
include 'Constant.lua'
include 'ConvertToVGGInput.lua'
include 'TrainableTensor.lua'
include 'TVLoss.lua'
include 'Provider.lua'
include 'Transformer.lua'
include 'Transformer2.lua'
include 'Transformer3.lua'
include 'Transformer4.lua'
include 'Transformer5.lua'
include 'TransformerAC.lua'
include 'TransformerAE.lua'
include 'TransformerAE2.lua'
include 'TGAN.lua'
include 'TGANRec.lua'
include 'TGANHiddenRec.lua'
include 'CondGAN.lua'

deepwarp.modules = {}
include 'modules/ImToGaussianPyr.lua'
include 'modules/GaussianPyrToLaplacianPyr.lua'
include 'modules/LaplacianPyrToIm.lua'
include 'modules/RandomWarp.lua'
include 'modules/PrintGrads.lua'
include 'modules/ScaleGrads.lua'
include 'modules/SpatialReplication.lua'
include 'modules/InvTanh.lua'
include 'modules/SplitBatch.lua'

deepwarp.base = {}
include 'CondDiscriminator.lua'
include 'CondDiscriminatorTmp.lua'
include 'ConvSrcMs.lua'
include 'ConvSrcMs2.lua'
include 'ConvSrcMs3.lua'
include 'ConvSrcMs4.lua'
include 'ConvSrcMs5.lua'
include 'ConvSrcMs5AC.lua'
include 'base/init.lua'

deepwarp.unsupervised = {}
include 'unsupervised/Provider.lua'
include 'unsupervised/DirectTransformer.lua'
include 'unsupervised/Transformer.lua'
include 'unsupervised/TGAN.lua'

deepwarp.unsupervised.base = {}
include 'unsupervised/TNet1.lua'
include 'unsupervised/TResNet.lua'
include 'unsupervised/TResNetDirect.lua'

deepwarp.vae_driven = {}
include 'vae_driven/Provider.lua'

deepwarp.vae = {}
include 'vae/GaussianCriterion.lua'
include 'vae/LaplacianCriterion.lua'
include 'vae/VGGCriterion.lua'
include 'vae/VGG2Criterion.lua'
include 'vae/ContentCriterion.lua'
include 'vae/KLDCriterion.lua'
include 'vae/GANDCriterion.lua'
include 'vae/GANGCriterion.lua'
include 'vae/GAN3WayDCriterion.lua'
include 'vae/GAN3WayGCriterion.lua'
include 'vae/Provider.lua'
include 'vae/PairsProvider.lua'
include 'vae/NNPairsProvider.lua'
include 'vae/Sampler.lua'
include 'vae/VAE.lua'
include 'vae/TVAE.lua'
include 'vae/TVAE2.lua'
include 'vae/TVAEPerceptual.lua'
include 'vae/TVAEPerceptual2.lua'
include 'vae/TVAEPerceptual3.lua'
include 'vae/DirectTransformer.lua'
include 'vae/AffineCorrectionTransformer.lua'
include 'vae/Abomination.lua'

deepwarp.ali = {}
include 'ali/Provider.lua'
include 'ali/Transformer.lua'
include 'ali/WarpTransformer.lua'
include 'ali/TALI.lua'

deepwarp.ali.base = {}
include 'ali/ResDirNet.lua'
include 'ali/ResWrpAllMixAllNet.lua'
include 'ali/BtlResWrpAllMixAllNet.lua'
include 'ali/BtlWrpAllMixAllNetOld.lua'
include 'ali/BtlWrpAllMixAllNet.lua'

deepwarp.laplacian = {}
include 'laplacian/WrpLapMixLowT.lua'
include 'laplacian/WrpAllMixLowT.lua'

deepwarp.laplacian.base = {}
include 'laplacian/BtlWrpLapMixLowNet.lua'
include 'laplacian/BtlWrpAllMixLowNet.lua'

deepwarp.jcjohnson = {}
include 'jcjohnson/Transformer.lua'

deepwarp.multistage = {}
include 'multistage/Transformer.lua'

deepwarp.multistage.base = {}
include 'multistage/base.lua'
include 'multistage/unet.lua'
include 'multistage/unet2.lua'

return deepwarp
