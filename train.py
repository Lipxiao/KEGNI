import os
import logging
from model import E2EModel
from dataloader import creat_KGEdataloader
from train import Trainer
from dataset import MAEDataset
import pandas as pd
import sys
from utils import  parser_args
import torch
import datetime
import numpy as np
import random

current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
# 日志文件名加上当前时间
log_filename = f"my_log_{formatted_time}.log"

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False

def main():
    set_seed(0)
    args = parser_args()
    # 创建日志记录器
    prefix, _ = os.path.splitext(os.path.basename(args.input))
    logger = logging.getLogger(prefix)
    logger.setLevel(logging.DEBUG)    
    log_folder = "log"
    os.makedirs(log_folder, exist_ok=True)
    # 设置日志文件的路径
    log_filename = os.path.join(log_folder, f"{prefix}_{formatted_time}.log")
    fh = logging.FileHandler(filename = log_filename)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("Command: " + " ".join(sys.argv))




    sc_dataset = MAEDataset(input = args.input, n_neighbors = args.n_neighbors)
    
    kge_file = args.data_path
    kge_data = pd.read_csv(kge_file,sep='\t',header = None)
    
    KGE_dataloader = creat_KGEdataloader(**args,kge_data = kge_data, sc_dataset = sc_dataset)
    kgg_kgg_iter,scg_scg_iter,scg_kgg_iter,kgg_scg_iter = KGE_dataloader.kgg_kgg_iter, KGE_dataloader.scg_scg_iter, KGE_dataloader.scg_kgg_iter, KGE_dataloader.kgg_scg_iter
    kgg2id, relation2id, scg2id = KGE_dataloader.kgg2id, KGE_dataloader.relation2id, KGE_dataloader.scg2id
    

    model = E2EModel(**args,num_features = sc_dataset.num_features,
                     kgg2id = kgg2id, relation2id = relation2id, scg2id = scg2id)

    trainer = Trainer(
        args = args,
        model=model,
        sc_dataset_iter=enumerate(sc_dataset),
        kgg_kgg_iter = kgg_kgg_iter,
        scg_scg_iter = scg_scg_iter,
        scg_kgg_iter = scg_kgg_iter,
        kgg_scg_iter = kgg_scg_iter,
        logger = logger
    )
    trainer.train()


if __name__ == "__main__":
    main()