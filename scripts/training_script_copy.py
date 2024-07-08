import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import modules
from configs.config import get_cfg_defaults
from nsdhandling.core import NsdData
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Script for training encoders"
    )

    parser.add_argument(
        '-s','--subj',
        type=int,
        help='Integer id of subject, e.g. 1'
    )

    parser.add_argument(
        '-f','--finetune',
        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), required=True,
        help='Flag to toggle bacbone finetuning, True will finetune backbone'
    )

    parser.add_argument(
        '-p','--percent',
        type=int,
        help='Percentage of total filters per layer to extract for readout'
    )

    parser.add_argument(
        '-c','--config',
        type=str,
        help='Config file name, if not done, please define config folder as environment variable named NSD_CONFIG_PATH'
    )

    args=parser.parse_args()
    print(args)

    #sort out the config file
    cfg=get_cfg_defaults()
    cfg.merge_from_file(cfg.PATHS.NSD_CONFIG+args.config)
    opts=["BACKBONE.FINETUNE",args.finetune,"BACKBONE.PERCENT_OF_CHANNELS",args.percent]
    cfg.merge_from_list(opts)
    cfg.freeze()
    print(cfg)

    #load the subject data
    nsd_data=NsdData([args.subj])
    nsd_data.load_preprocessed(cfg.BACKBONE.INPUT_SIZE)
    nsd_data.make_data_loaders(batch_size=cfg.TRAIN.BATCH_SIZE)

    #get the encoder
    #Nv=int(nsd_data.data[0]['Nv']) #number of voxels
    #enc=modules.LitEncoder(cfg,nsd_data.data_loaders_train[0])

    #fit the model
    #trainer = modules.EncoderTrainer(cfg,max_epochs=cfg.TRAIN.MAX_EPOCHS)
    #trainer.fit(model=enc.cuda(), train_dataloaders=enc.encoder.data_loader)

if __name__ == "__main__":
    main()
