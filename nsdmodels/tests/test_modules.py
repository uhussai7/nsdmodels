import pytest
import torch
import sys
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import modules


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


