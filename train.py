import os
import logging
from model import KEGNI
from dataloader import KGEdataloader
from train import Trainer
from dataset import MAEDataset
import pandas as pd
import sys
from utils import parser_args
import torch
import datetime
import numpy as np
import random
from utils import set_seed
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")

log_filename = f"my_log_{formatted_time}.log"

def main():
    set_seed(42)
    args = parser_args()

    prefix, _ = os.path.splitext(os.path.basename(args.input))
    logger = logging.getLogger(prefix)
    logger.setLevel(logging.DEBUG)
    log_folder = "log"
    os.makedirs(log_folder, exist_ok=True)

    log_filename = os.path.join(log_folder, f"{prefix}_{formatted_time}.log")
    fh = logging.FileHandler(filename=log_filename)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("Command: " + " ".join(sys.argv))

    sc_dataset = MAEDataset(input=args.input, n_neighbors=args.n_neighbors)

    kge_dataloader = KGEdataloader(sc_dataset=sc_dataset,**vars(args))
    
    if kge_dataloader.kgg_kgg_triples:
        kgg_kgg_iter = kge_dataloader.kgg_kgg_dataloader()#TODO 可以考虑删除kgg_kgg,kgg_scg
    else:
        kgg_kgg_iter = None
        
    if kge_dataloader.scg_scg_triples:
        scg_scg_iter = kge_dataloader.scg_scg_dataloader()
    else:
        scg_scg_iter = None
        
    if kge_dataloader.scg_kgg_triples:
        scg_kgg_iter = kge_dataloader.scg_kgg_dataloader()
    else:
        scg_kgg_iter = None
        
    if kge_dataloader.kgg_scg_triples:
        kgg_scg_iter = kge_dataloader.kgg_scg_dataloader()
    else:
        kgg_scg_iter = None       


    model = KEGNI(num_features=sc_dataset.num_features,
                  kgg2id=kge_dataloader.kgg2id,
                  relation2id=kge_dataloader.relation2id,
                  scg2id=kge_dataloader.scg2id,
                  **vars(args))

    trainer = Trainer(
        args=args,
        model=model,
        sc_dataset_iter=enumerate(sc_dataset),
        kgg_kgg_iter=kgg_kgg_iter,
        scg_scg_iter=scg_scg_iter,
        scg_kgg_iter=scg_kgg_iter,
        kgg_scg_iter=kgg_scg_iter,
        logger=logger
    )
    trainer.train()


if __name__ == "__main__":
    main()
