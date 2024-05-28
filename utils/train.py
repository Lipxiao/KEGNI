import os
import logging
from train.models import E2EModel
from train.trainer import Trainer
from train.dataset import KGEDataset, MAEDataset
from torch.utils.data import DataLoader
import pandas as pd
import sys
from args import  parser_args
from kge_model.dataloader import BidirectionalOneShotIterator
import torch
import pickle as pkl

import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
# 日志文件名加上当前时间
log_filename = f"my_log_{formatted_time}.log"
import numpy as np
import random
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if torch.cuda.is_available() :
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


    if args.save_processed:
        protein_seq_dataset = None
        protein_seq_dataset = MAEDataset()
        kge_file = args.data_path
        # kge_file = '/media/disk/project/KnowledgeGraphEmbedding/data/KEGG/KEGG_mHSC-E_1.tsv'
        kge_data = pd.read_csv(kge_file,sep='\t',header = None)

        # kge_file = 'nomolecular/PathwayCommons12.All.hgnc.nomolecular.sif'
        # kge_file = 'KEGG/KEGG_all_mm_filter.tsv'  
        # kge_file = 'KEGG/KEGG_all_hs_filter.tsv'  
        # kge_file = 'TRRUST/trrust_mouse_filter.tsv'
        # kge_file = 'TRRUST/trrust_human.tsv'
        # kge_file = 'TRRUST/trrust_mouse.tsv'
        # kge_file = 'KEGG/combine_hs_filter.tsv'
        # kge_file = 'reactome/reactome_hs.tsv'  #有三种类型的实体,转录组中的基因,非转录组中的基因,pathway
        # kge_data = pd.read_csv(os.path.join(args.data_path, kge_file),sep='\t',header = None)
        
        kge_data[0] = kge_data[0].str.upper()
        kge_data[2] = kge_data[2].str.upper()
        entity = (set(kge_data[0].tolist() + kge_data[2].tolist()))-set(protein_seq_dataset.node2id.keys())
        # entity2id = {string: index for index, string in enumerate(entity)}
        # relation2id = {string: index for index, string in enumerate(set(kge_data[1]))}
        entity2id = {string: index for index, string in enumerate(sorted(list(entity)))}
        relation2id = {string: index for index, string in enumerate(sorted(list(set(kge_data[1]))))}
        protein2id = protein_seq_dataset.node2id
        go_go_triples = []
        protein_go_triples = []
        protein_protein_triples = []
        go_protein_triples = []
   
        for index, row in kge_data.iterrows():
            h, r, t = row[0], row[1], row[2]
            if ((h in entity) & (t in entity)):
            # triples.append((entity2id[h], relation2id[r], entity2id[t]))
                go_go_triples.append((entity2id[h], relation2id[r], entity2id[t]))
            elif (h in protein2id) & (t in protein2id):
                protein_protein_triples.append((protein2id[h],relation2id[r],protein2id[t]))
            elif ((h in protein2id) &  (t in entity)):
                protein_go_triples.append((protein2id[h],relation2id[r],entity2id[t]))
            # elif ((h in entity) &  (t in protein2id)):
            #     protein_go_triples.append((protein2id[t],relation2id[r],entity2id[h]))                
            elif ((h in entity) &  (t in protein2id)):
                go_protein_triples.append((entity2id[h],relation2id[r],protein2id[t]))   
                
        # with open(prefix+'_kge_processed_data.pkl', 'wb') as f:
            # pkl.dump((go_go_triples, protein_protein_triples, protein_go_triples,go_protein_triples,protein2id,relation2id,entity2id,protein_seq_dataset), f)
    # if args.load_processed:
        # with open(prefix+'_kge_processed_data.pkl', 'rb') as f:
            # go_go_triples, protein_protein_triples, protein_go_triples,go_protein_triples,protein2id,relation2id,entity2id,protein_seq_dataset = pkl.load(f)

    go_go_iter= None  
    protein_go_iter = None
    protein_protein_iter = None
    go_protein_iter = None
    
    if go_go_triples:
        go_go_dataloader_head = DataLoader(
            KGEDataset(go_go_triples,  negative_sample_size = args.negative_sample_size,
                        mode = 'head-batch', nentity = len(entity2id)), 
            batch_size=int(args.batch_size),
            # batch_size = len(go_go_triples),
            shuffle=True, 
            # num_workers=max(1, args.cpu_num//2),
            collate_fn=KGEDataset.collate_fn
        )
    
        go_go_dataloader_tail = DataLoader(
            KGEDataset(go_go_triples,  negative_sample_size = args.negative_sample_size, 
                        mode = 'tail-batch',nentity = len(entity2id)), 
            batch_size=int(args.batch_size),
            # batch_size = len(go_go_triples),
            shuffle=True, 
            # num_workers=max(1, args.cpu_num//2),
            collate_fn=KGEDataset.collate_fn
        )
        go_go_iter = BidirectionalOneShotIterator(go_go_dataloader_head, go_go_dataloader_tail)
        # go_go_iterator = BidirectionalOneShotIterator.one_shot_iterator( go_go_dataloader_tail)
    if protein_protein_triples:
        protein_protein_dataloader_head = DataLoader(
            KGEDataset(protein_protein_triples, negative_sample_size = args.negative_sample_size, 
                        mode = 'head-batch',nentity = len(protein2id)), 
            batch_size=int(args.batch_size),
            shuffle=True, 
            # num_workers=max(1, args.cpu_num//2),
            collate_fn=KGEDataset.collate_fn
        )
        protein_protein_dataloader_tail = DataLoader(
            KGEDataset(protein_protein_triples,negative_sample_size = args.negative_sample_size,
                        mode = 'tail-batch', nentity = len(protein2id)), 
            batch_size=int(args.batch_size),
            shuffle=True, 
            # num_workers=max(1, args.cpu_num//2),
            collate_fn=KGEDataset.collate_fn
        )
        protein_protein_iter = BidirectionalOneShotIterator(protein_protein_dataloader_head,protein_protein_dataloader_tail)
        # protein_protein_iter = BidirectionalOneShotIterator.one_shot_iterator(protein_protein_dataloader_tail)
        
    if protein_go_triples:
        protein_go_dataloader_head = DataLoader(
            KGEDataset(protein_go_triples,  negative_sample_size = args.negative_sample_size,
                            mode = 'head-batch', nentity= len(protein2id)), 
            batch_size=int(args.batch_size),
            # batch_size = len(protein_go_triples),
            shuffle=True, 
            # num_workers=max(1, args.cpu_num//2),
            collate_fn=KGEDataset.collate_fn
        )
        protein_go_dataloader_tail = DataLoader(
            KGEDataset(protein_go_triples,  negative_sample_size = args.negative_sample_size,
                            mode = 'tail-batch', nentity= len(entity2id)), 
            batch_size=int(args.batch_size),
            # batch_size = len(protein_go_triples),
            shuffle=True, 
            # num_workers=max(1, args.cpu_num//2),
            collate_fn=KGEDataset.collate_fn
        )
        # protein_go_iter =BidirectionalOneShotIterator(protein_go_dataloader_head,protein_go_dataloader_tail)
        protein_go_iter =BidirectionalOneShotIterator.one_shot_iterator(protein_go_dataloader_tail)
    if go_protein_triples:    
        go_protein_dataloader_head = DataLoader(
            KGEDataset(go_protein_triples,  negative_sample_size = args.negative_sample_size,
                            mode = 'head-batch', nentity= len(entity2id)), 
            batch_size=int(args.batch_size),
            # batch_size = len(protein_go_triples),
            shuffle=True, 
            # num_workers=max(1, args.cpu_num//2),
            collate_fn=KGEDataset.collate_fn
        )    
        go_protein_dataloader_tail = DataLoader(
            KGEDataset(go_protein_triples,  negative_sample_size = args.negative_sample_size,
                            mode = 'tail-batch', nentity= len(protein2id)), 
            batch_size=int(args.batch_size),
            # batch_size = len(protein_go_triples),
            shuffle=True, 
            # num_workers=max(1, args.cpu_num//2),
            collate_fn=KGEDataset.collate_fn
        )   
        go_protein_iter =BidirectionalOneShotIterator.one_shot_iterator(go_protein_dataloader_head)
        # go_protein_iter =BidirectionalOneShotIterator(go_protein_dataloader_head,go_protein_dataloader_tail)


    model = E2EModel(num_features = protein_seq_dataset.num_features,
                     protein2id = protein2id, relation2id = relation2id, entity2id = entity2id)

    trainer = Trainer(
        model=model,
        sc_dataset_iter=enumerate(protein_seq_dataset),
        kgg_kgg_iter=go_go_iter,
        scg_kgg_iter = protein_go_iter,
        scg_scg_iter = protein_protein_iter,
        kgg_scg_iter = go_protein_iter,
        logger = logger
    )
    trainer.train()


if __name__ == "__main__":
    main()