#this is for tests for clip

import clip
from configs.config import get_cfg_defaults
from nsdhandling.core import NsdData
from models import modules
import torch
from nltk.corpus import words
from torch.utils.data import TensorDataset,DataLoader

def test_tokens():
    """
    Test token related actions
    """
    
    #config
    cfg=get_cfg_defaults()
    cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+'/clip.yaml')
    cfg.freeze()
    print(cfg)

    #get the clip model
    model=torch.load(cfg.PATHS.BACKBONE_FILES+cfg.BACKBONE.FILE)
    
    #get words
    common_words=words.words()
    print('Tokenizing words..')
    tokens=clip.tokenize(common_words)

    return model,tokens


def test_text_encoder():
    """
    Test text encoder 
    """

    #config
    cfg=get_cfg_defaults()
    cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+'/clip_text.yaml')
    cfg.freeze()
    print(cfg)

    #get encoder
    enc=modules.Encoder(cfg,Nv=100,Nt=640)

    return enc

    