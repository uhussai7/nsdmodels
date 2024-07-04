#imports
from torch.nn import Module, Parameter, ParameterList, Linear, init
from torch.nn.functional import mse_loss
from torch import matmul,optim
import torch
import math
import os
from torchvision.models.feature_extraction import create_feature_extractor
from .utils import map_times_rf,layer_shapes,unique_2d_layer_shapes, channels_to_use,channel_summer
import lightning as L
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.optim import Adam, SGD
from pathlib import Path

class EncoderTrainer(L.Trainer):
    """
    Class for training the encoder
    """
    def __init__(self,cfg,subj=1,*args,**kwargs): #we do per subject
        self.cfg=cfg
        self.subj=subj
        super().__init__(*args,
                        default_root_dir=self.create_dir_name(),**kwargs)

    def create_dir_name(self): # we need to make sure all the models are in the right places
    #the main hyperparameters are backbone-name, finetune or not, percent of filter
    #we will do seperate models for individuals for now, so subject id is also needed
        self.out_path=Path(os.path.join(self.cfg.PATHS.NSD_ENCODER,'subj%02d'%self.subj,
                          self.cfg.BACKBONE.NAME,
                          'finetune-'+str(self.cfg.BACKBONE.FINETUNE),
                          'percent_channels-%d'%(int(self.cfg.BACKBONE.PERCENT_OF_CHANNELS))))
        self.out_path.mkdir(parents=True,exist_ok=True)
        return str(self.out_path)

class LitEncoder(L.LightningModule):
    """
    Lightning wrapper for Encoder class
    """
    def __init__(self,cfg,data_loader):        
        super().__init__()  
        self.encoder=Encoder(cfg,data_loader)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_hat = self.encoder(x)
        loss = mse_loss(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        #print(loss)
        self.log("train_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        return torch.stack((self.encoder(x),y))

    def configure_optimizers(self):
        if self.encoder.cfg.BACKBONE.FINETUNE == True: #train backbone or not
            optimizer = Adam(params=self.parameters(), lr=0.0001)
        else:
            params_to_pass=[
                            {'params': self.encoder.readout.rfs.parameters()},
                            {'params': self.encoder.readout.w},
                            {'params': self.encoder.readout.b}
                            ]
            optimizer = Adam(params=params_to_pass, lr=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=[2, 3, 4, 5, 6, 7], gamma=0.8)
        return [optimizer], [scheduler]

    def forward(self,x):
        return self.encoder(x)

class EncoderReadOut(Module):
    """
    Module for fmri encoder readout
    """
    def __init__(self,cfg,feature_extractor,
                 rfs, #instance of ReceptiveField
                 N_channels):
        #TODO: need descriptor for this __init__ function
        super().__init__()
        self.cfg=cfg
        self.feature_extractor=feature_extractor
        self.rfs=rfs
        self.N_channels=N_channels
        self.Nv = self.rfs.Nv

        self.w=Parameter(torch.empty([self.N_channels,self.Nv]))
        self.b=Parameter(torch.empty(self.Nv))

        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.b)
        init.kaiming_uniform_(self.w,a=math.sqrt(5))

    def forward(self,x):
        features=self.feature_extractor(x)
        sigma=self.rfs(features) #shape [B,N_channels,Nv]
        return (sigma[:,:,:]*self.w[None,:,:]).sum(-2) + self.b #summing the channel dimension here

class Encoder(Module): 
    """
    Module for fmri encoder
    """
    def __init__(self,cfg,data_loader,imgs=None):
        #TODO: need descriptor for this __init__ function and some comments below
        super().__init__()
        self.cfg=cfg
        self.image_features=ImageFeatures(cfg)
        self.get_rf_sizes()
        self.imgs=imgs
        self.data_loader=data_loader
        self.get_channel_basis(self.get_imgs()) #TODO: this is done without cuda at initialization
        self.Nv=list(data_loader)[0][1].shape[-1]
        self.rfs=ReceptiveField(self.Nv,self.rf_sizes,self.layer_to_rf_size,self.channel_basis)
        self.readout=EncoderReadOut(cfg,self.image_features,self.rfs,self.N_channels)

    def get_imgs(self,N_imgs=None):
        """
        Extracts images from dataloader if self.imgs is None
        """
        if self.imgs is not None:
            return self.imgs
        else:
            return torch.cat([ _[0] for _ in list(self.data_loader)])

    def get_rf_sizes(self):
        """Gets the unique rf-sizes to make receptive fields"""
        self.layer_shps=layer_shapes(self.image_features,self.cfg.BACKBONE.INPUT_SIZE)
        (
            self.rf_sizes, #unique rf_sizes
            self.layer_to_rf_size, #index of rf_size for each layer 
            self.channels #channels in each layer

        ) = unique_2d_layer_shapes(self.cfg.BACKBONE.LAYERS_TO_EXTRACT,self.layer_shps)
        
    def get_channel_basis(self,imgs):
        self.channel_basis=channels_to_use(self.cfg.BACKBONE.LAYERS_TO_EXTRACT,
                                           self.image_features,
                                           imgs,
                                           self.cfg.BACKBONE.PERCENT_OF_CHANNELS)
        self.N_channels=channel_summer(self.channel_basis)

    def forward(self,x):
        return self.readout(x)

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
        try:
            print('Loading backbone from:', self.backbone_file)
            self.backbone=torch.load(self.backbone_file)
            if isinstance(self.backbone,Module):
                print('Module loaded')
        except:
            print('Failed to load backbone')

    def get_feature_extractor(self):
        """Creates feature extractor using torchvision's feature_extraction module"""
        self.feature_extractor=create_feature_extractor(self.backbone,return_nodes=self.cfg.BACKBONE.LAYERS_TO_EXTRACT)

    def forward(self,x):
        return self.feature_extractor(x)

class ReceptiveField(Module):
    """
    Module for a receptive field, takes activative maps and multiplies with rf-fields for each voxel
    """
    def __init__(self,Nv,rf_sizes,layer_to_rf_size,channel_basis):
        """
        Initialization function
        TODO: fix this comment section
        Args:
            Nv: number of voxels
            rf_sizes: list of rf sizes [(H_1,W_1), (H_2,W_2), (H_3,W_3), ...]
        """
        super().__init__()
        self.Nv=Nv
        self.rf_sizes=rf_sizes
        self.rfs=ParameterList([Parameter(torch.empty((Nv,)+hw)) for hw in self.rf_sizes])
        self.layer_to_rf_size=layer_to_rf_size
        self.channel_basis=channel_basis

        self.reset_parameters()

    def reset_parameters(self):
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
            x_=x_[:,self.channel_basis[key],:,:]
            rf=self.rfs[self.layer_to_rf_size[key]]
            out.append(map_times_rf(x_,rf))
        return torch.cat(out,dim=1)





