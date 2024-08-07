import pytest
import torch
import sys
import os 
from torch.utils.data import Subset, DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import modules
from configs.config import get_cfg_defaults
from nsdhandling.core import NsdData
import numpy as np
import lightning as L
import matplotlib.pyplot as plt

def test_rf():
    """
    Simple test for rf sizes
    """
    Nv=10 #voxels
    B=4 #batch
    Cs=[4,8,16] #layer channels
    rf_sizes=((128,128),(64,64),(32,32)) #layer resolutions
    x=[torch.rand((B,C,)+rf_size) for C,rf_size in zip(Cs,rf_sizes)] #fake activations
    print('The activation maps have sizes:')
    [print(x_.shape) for x_ in x]
    rf = modules.ReceptiveField(Nv,rf_sizes)
    out=rf(x)
    assert out.shape == (B,sum(Cs),Nv)
    print('No shape inconsistency, output has shape: ',out.shape )

def test_training_prediction(N=10000,N_val=1000,batch_size=1,max_epochs=10):

    #load nsddata for one subject 
    #use this lightning thing to invoke encoder
    #see if it trains and all weights are updating (important)

    #get the default config
    cfg=get_cfg_defaults()


    #Loading the data
    nsd_data=NsdData([1])
    nsd_data.load_preprocessed(cfg.BACKBONE.INPUT_SIZE)

    #handle text or not for data_loaders
    if cfg.BACKBONE.TEXT == True:
        import clip
        nsd_data.make_data_loaders(batch_size=cfg.TRAIN.BATCH_SIZE,text=True,tokenizer=clip.tokenize)
    else:
        nsd_data.make_data_loaders(batch_size=cfg.TRAIN.BATCH_SIZE)

    #use subset dataset
    data_loader_train = DataLoader(
                                    Subset(nsd_data.data_loaders_train[0].dataset,np.arange(0,N)),
                                    batch_size=batch_size
                                  )
    
    #get an encoder
    Nv=nsd_data.data_loaders_train[0].dataset.tensors[0]
    enc=modules.LitEncoder(cfg,data_loader_train,imgs=nsd_data.data_loaders_train[0].dataset.tensors[0][:10000])

    #fit the model
    trainer = modules.EncoderTrainer(cfg,1,max_epochs=max_epochs)
    trainer.fit(model=enc.cuda(), train_dataloaders=enc.encoder.data_loader)

    #make predictions on validation dataset
    #single
    data_loader_val_single = DataLoader(
                                Subset(nsd_data.data_loaders_val_single[0].dataset,np.arange(0,N_val)),
                                batch_size=batch_size
                                )
    predictions_single=torch.cat(trainer.predict(enc,dataloaders=data_loader_val_single),dim=1)
    corr_single=torch.asarray([torch.corrcoef(predictions_single[:,:,i])[0,1] for i in range(0,predictions_single.shape[-1])])

    #multi
    data_loader_val_multi = DataLoader(
                                Subset(nsd_data.data_loaders_val_multi[0].dataset,np.arange(0,N_val)),
                                batch_size=batch_size
                                )
    predictions_multi=torch.cat(trainer.predict(enc,dataloaders=data_loader_val_multi),dim=1)
    corr_multi=torch.asarray([torch.corrcoef(predictions_multi[:,:,i])[0,1] for i in range(0,predictions_multi.shape[-1])])

    #save a correlation histogram in the log_dir of the trainer
    fig,ax=plt.subplots()
    ax.hist(corr_single,100)
    plt.savefig(trainer.log_dir+'/corr_single_hist.png')    

    fig,ax=plt.subplots()
    ax.hist(corr_multi,100)
    plt.savefig(trainer.log_dir+'/corr_multi_hist.png') 

    return enc,nsd_data,trainer,predictions_single,corr_single,corr_multi


def test_nsddata():
    """
    Test for loading data
    """
    nsd_data=NsdData([1])
    nsd_data.load_preprocessed()

    print('Done')
    print('From the data_loaders the train dataset has shapes:',nsd_data.data_loaders_train[0].dataset.tensors[0].shape,
                                                                nsd_data.data_loaders_train[0].dataset.tensors[1].shape)

def test_trainer():
    """
    Test for EncoderTrainer class
    """
    
    #test outpath
    cfg=get_cfg_defaults()
    trainer=modules.EncoderTrainer(cfg)
    return trainer

def test_encoder():
    """
    Simple test for encoder shapes
    """
    cfg=get_cfg_defaults()

    #parameters for the data
    Nv=10
    Nimgs=100
    imgs=torch.rand([Nimgs,3,256,256])
    fmri=torch.rand([Nimgs,Nv])
    data_loader=DataLoader(TensorDataset(imgs,fmri))
    enc=modules.Encoder(cfg,data_loader) 
    print('Loaded successfully')
    batch_size=3
    out=enc(imgs[0:batch_size])
    assert out.shape == (batch_size,Nv)
    print('No shape inconsistency, output has shape: ',out.shape )
