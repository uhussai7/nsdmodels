#imports
from torch.nn import Module, Parameter, ParameterList, init, Linear
from torch import matmul
import torch
import math
import os
from torchvision.models.feature_extraction import create_feature_extractor
from utils improt map_times_rf,layer_shapes,unique_2d_layer_shapes, channels_to_use


# !! The encoder shouldn't deal with channel sorting etc, it should be procded these things
# beta= w_I <phi^I | rho > 
# Encoder needs 
# - rf_sizes
# - feature extractor
# - feature extractor layer to rf_size mapping
# - total number of channels
# - channels to take (channel basis)
# - total number of voxels

class EncoderReadOut(Module):
    """
    Module for fmri encoder
    """
    def __init__(self,cfg,feature_extractor,
                 rfs, #instance of ReceptiveField
                 layer_to_rf_size,
                 N_channels,
                 channel_basis)
        super().__init__()
        self.cfg=cfg
        self.feature_extractor=feature_extractor
        self.rfs=rfs
        self.layer_to_rf_size=layer_to_rf_size
        self.N_channels=self.N_channels
        self.channel_basis=self.channel_basis
        self.Nv = self.rfs.Nv

        self.w=Parameter(torch.empty([self.N_channels,self.Nv]))
        self.b=Parameter(torch.empty(Nv))

        self.reset_parameters()

      def reset_parameters(self):
        init.uniform_(self.b)
        init.kaiming_uniform_(self.w,a=math.sqrt(5))

    def forward(self,x):
        features=self.feature_extractor(x)
        sigma=self.rfs(features) #shape [B,N_channels,Nv]
        return (sigma[:,:,:]*self.w[None,:,:]).sum(-2) + self.b
        
        

# class Encoder(Module): 
#     """
#     Module for fmri encoder
#     """
#     def __init(self,cfg,imgs):
#         self.cfg=cfg
#         self.image_features=ImageFeatures(cfg)

#     def get_rf_sizes(self):
#         """Gets the unique rf-sizes to make receptive fields"""
#         self.layer_shps=layer_shapes(self.image_features,self.cfg.BACKBONE.INPUT_SIZE)
#         (
#             self.rf_sizes, #unique rf_sizes
#             self.layer_to_rf_size, #index of rf_size for each layer 
#             self.channels #channels in each layer

#         ) = unique_2d_layer_shapes(self.cfg.BACKBONE.LAYERS_TO_EXTRACT,self.layer_shps)
        
#     def get_channel_basis(self,imgs):
#         self.channel_basis=channels_to_use(self.cfg.BACKBONE.LAYERS_TO_EXTRACT,
#                                            self.image_features,
#                                            imgs,
#                                            self.cfg.BACKBONE.PERCENT_OF_CHANNELS)


class ImageFeatures(Module):
    """
    Module to extract features from a backbone
    """
    def __init__(self,cfg):
        """
        Intialization function for ImageFeatures class

        Args:
            cfg: yacs configuration node
        """
        super().__init__()
        self.cfg=cfg
        self.get_backbone()
        self.get_feature_extractor()

    def get_backbone(self):
        """Loads backbone architecture and weights (from one file)"""
        self.backbone_file=os.path.join(self.cfg.PATHS.BACKBONE_FILES + self.cfg.BACKBONE.FILE)
        self.backbone=torch.load(self.backbone_file)

    def get_feature_extractor(self):
        """Creates feature extractor using torchvision's feature_extraction module"""
        self.feature_extractor=create_feature_extractor(backbone,return_nodes=self.cfg.BACKBONE.LAYERS_TO_EXTRACT)

    def forward(self,x):
        return self.feature_extractor(x)

class ReceptiveField(Module):
    """
    Module for a receptive field, takes activative maps and multiplies with rf-fields for each voxel
    """
    def __init__(self,Nv,rf_sizes,layer_to_rf_size):
        """
        Initialization function

        Args:
            Nv: number of voxels
            rf_sizes: list of rf sizes [(H_1,W_1), (H_2,W_2), (H_3,W_3), ...]
        """
        super().__init__()
        self.Nv=Nv
        self.rf_sizes=rf_sizes
        self.rfs=ParameterList([Parameter(torch.empty((Nv,)+hw)) for hw in self.rf_sizes])
        self.b=Parameter(torch.empty(self.Nv))
        self.layer_to_rf_size=layer_to_rf_size

        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.b)
        for rf in self.rfs:
            init.kaiming_uniform_(rf,a=math.sqrt(5))

    def forward(self,x):
        """
        Forward function for ReceptiveField class

        Args:
            x: output of ImageFeatures class
        Returns:
            tensor: Tensor with size [B,Nv,C_0+C_1+...] = [B,Nv,C_total]
        """
        out=[]
        for key,x_ in x.items():
            rf=self.rfs[self.layer_to_rf_size[key]]
            out.append(map_times_rf(x_,rf))
        return torch.cat(out,dim=1)





