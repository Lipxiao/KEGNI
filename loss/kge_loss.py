from math import gamma
import os
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import torch.nn as nn
# from torch.nn import CrossEntropyLoss
from torch.nn.modules.sparse import Embedding
from transformers.file_utils import ModelOutput
# from transformers.utils import logging
import logging
from transformers.deepspeed import is_deepspeed_zero3_enabled
import collections

# from src.args import  parser_args,parser_kgeargs,parser_maeargs
# from args import  parser_args
from model.models import E2EModel

# from kge_model.utils import parse_args

# logger = logging.get_logger(__name__)


class KGEloss:
    """
    Loss function for KE.
    """
    def __init__(self,args):
        self.args = args

    def __call__(
        self,
        model: E2EModel,
        embedding = None,
        kgg_kgg_inputs = None,
        scg_kgg_inputs= None,
        kgg_scg_inputs= None,
        scg_scg_inputs= None,
        **kwargs
    ):

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }
      
        self.gamma = model.kg_model.gamma
        self.embedding_range = model.kg_model.embedding_range
        # self.gamma = self.kge_args.gamma
        
        if self.args.model not in model_func:
            raise ValueError('model %s not supported' % self.args.model)
        
        if self.args.device < 0:
            device = "cpu"
        else:
            device = f"cuda:{self.args.device}" if torch.cuda.is_available() else "cpu"

        # model.eval()
        total_loss = torch.tensor(0.0).to(device)
        all_loss = collections.defaultdict(float)
        

        if  scg_kgg_inputs:
            positive_sample, negative_sample, subsampling_weight, mode = scg_kgg_inputs
            # subsampling_weight = torch.tensor(1.0)
            subsampling_weight = subsampling_weight.to(device)
            batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
            if mode == "tail-batch":
                head_ids = [list[0] for list in positive_sample]
                relations_ids = [list[1] for list in positive_sample]
                head_embed, _,relation_embed= model(
                    embedding = embedding,
                    protein_ids = head_ids,
                    relation_ids = relations_ids,
                )     
                head_embed = head_embed.unsqueeze(1)    
                relation_embed = relation_embed.unsqueeze(1)                                                
                if positive_sample is not None:
                    tail_ids = [list[2] for list in positive_sample]
                    _, tail_embed ,_= model(
                        go_ids = tail_ids
                    )
                    tail_embed = tail_embed.unsqueeze(1)
                    score = model_func[self.args.model]( head = head_embed, 
                                                            relation = relation_embed,
                                                            tail = tail_embed,
                                                            mode = "tail-batch")
                    ke_loss = F.logsigmoid(score).squeeze(dim = 1)
                    pos_ke_loss = (- (subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

                if negative_sample is not None:
                    tail_ids = negative_sample.reshape(-1)
                    _,  tail_embed ,_= model(
                        go_ids = tail_ids
                        )      
                    tail_embed = tail_embed.reshape(batch_size, negative_sample_size, -1)
                    score = model_func[self.args.model]( head = head_embed, 
                                                            relation = relation_embed,
                                                            tail = tail_embed,
                                                            mode = "tail-batch")
                    ke_loss = F.logsigmoid(-score).mean(dim = 1)
                    neg_ke_loss = (- (subsampling_weight * ke_loss).sum()/subsampling_weight.sum()) 
            if mode == "head-batch":
                tail_ids = [list[2] for list in positive_sample]
                relations_ids = [list[1] for list in positive_sample]
                _,tail_embed,relation_embed= model(
                    go_ids = tail_ids,
                    relation_ids = relations_ids,
                    # protein_seq_inputs = protein_seq_inputs
                )    
                tail_embed = tail_embed.unsqueeze(1)
                relation_embed = relation_embed.unsqueeze(1)         
                if positive_sample is not None:
                    head_ids = [list[0] for list in positive_sample]
                    head_embed,_, _= model(
                        embedding = embedding,
                        protein_ids = head_ids, 
                                           )                    
                    head_embed = head_embed.unsqueeze(1)
                    score = model_func[self.args.model]( head = head_embed, 
                                                            relation = relation_embed,
                                                            tail = tail_embed,
                                                            mode = "head-batch")
                    ke_loss = F.logsigmoid(score).squeeze(dim = 1)
                    pos_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

                if negative_sample is not None:
                    head_ids = negative_sample.reshape(-1)
                    # tail_embed = self.onto_model.go_embedding[tail_ids].reshape(batch_size, negative_sample_size, -1)  
                    head_embed, _,_= model(
                        embedding = embedding,
                        protein_ids = head_ids, 
                    )                       
                    head_embed = head_embed.reshape(batch_size, negative_sample_size, -1)             
                    score = model_func[self.args.model]( head = head_embed, 
                                                            relation = relation_embed,
                                                            tail = tail_embed,
                                                            mode = "head-batch")
                    ke_loss = F.logsigmoid(-score).squeeze(dim = 1)
                    neg_ke_loss = (- (subsampling_weight * ke_loss.mean(dim = 1)).sum()/subsampling_weight.sum())
            protein_go_loss = (pos_ke_loss + neg_ke_loss)/2
            total_loss += protein_go_loss
            all_loss['protein_go_loss'] = protein_go_loss

        if  kgg_scg_inputs:
            positive_sample, negative_sample, subsampling_weight, mode = kgg_scg_inputs
            subsampling_weight = subsampling_weight.to(device)
            batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)        
            if mode == "tail-batch":
                head_ids = [list[0] for list in positive_sample]
                relations_ids = [list[1] for list in positive_sample]
                _,head_embed,relation_embed= model(
                    go_ids = head_ids,
                    relation_ids = relations_ids,
                )    
                head_embed = head_embed.unsqueeze(1)
                relation_embed = relation_embed.unsqueeze(1)         
                if positive_sample is not None:
                    tail_ids = [list[2] for list in positive_sample]
                    tail_embed,_, _= model(                    
                            embedding = embedding,
                            protein_ids = tail_ids)                    
                    tail_embed = tail_embed.unsqueeze(1)
                    score = model_func[self.args.model]( head = head_embed, 
                                                            relation = relation_embed,
                                                            tail = tail_embed,
                                                            mode = "tail-batch")
                    ke_loss = F.logsigmoid(score).squeeze(dim = 1)
                    pos_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

                if negative_sample is not None:
                    tail_ids = negative_sample.reshape(-1)
                    # tail_embed = self.onto_model.go_embedding[tail_ids].reshape(batch_size, negative_sample_size, -1)  
                    tail_embed, _,_= model(
                            embedding = embedding,
                            protein_ids = tail_ids)                       
                    tail_embed = tail_embed.reshape(batch_size, negative_sample_size, -1)             
                    score = model_func[self.args.model]( head = head_embed, 
                                                            relation = relation_embed,
                                                            tail = tail_embed,
                                                            mode = "tail-batch")
                    ke_loss = F.logsigmoid(-score).squeeze(dim = 1)
                    neg_ke_loss = (- (subsampling_weight * ke_loss.mean(dim = 1)).sum()/subsampling_weight.sum())
            
            if mode == "head-batch":
                tail_ids = [list[2] for list in positive_sample]
                relations_ids = [list[1] for list in positive_sample]
                tail_embed,_,relation_embed= model(
                    embedding = embedding,
                    protein_ids = tail_ids,
                    relation_ids = relations_ids,
                    # protein_seq_inputs = protein_seq_inputs
                )    
                tail_embed = tail_embed.unsqueeze(1)
                relation_embed = relation_embed.unsqueeze(1)         
                if positive_sample is not None:
                    head_ids = [list[0] for list in positive_sample]
                    _,head_embed, _= model(go_ids = head_ids, 
                                        #    protein_seq_inputs = protein_seq_inputs
                                           )                    
                    head_embed = head_embed.unsqueeze(1)
                    score = model_func[self.args.model]( head = head_embed, 
                                                            relation = relation_embed,
                                                            tail = tail_embed,
                                                            mode = "head-batch")
                    ke_loss = F.logsigmoid(score).squeeze(dim = 1)
                    pos_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

                if negative_sample is not None:
                    head_ids = negative_sample.reshape(-1)
                    # tail_embed = self.onto_model.go_embedding[tail_ids].reshape(batch_size, negative_sample_size, -1)  
                    _, head_embed,_= model(go_ids = head_ids)                       
                    head_embed = head_embed.reshape(batch_size, negative_sample_size, -1)             
                    score = model_func[self.args.model]( head = head_embed, 
                                                            relation = relation_embed,
                                                            tail = tail_embed,
                                                            mode = "head-batch")
                    ke_loss = F.logsigmoid(-score).squeeze(dim = 1)
                    neg_ke_loss = (- (subsampling_weight * ke_loss.mean(dim = 1)).sum()/subsampling_weight.sum())
            go_protein_loss = (pos_ke_loss + neg_ke_loss)/2
            # total_loss += go_protein_loss
            all_loss['go_protein_loss'] = go_protein_loss
                        
        if  scg_scg_inputs:
            positive_sample, negative_sample, subsampling_weight, mode = scg_scg_inputs
            # subsampling_weight = torch.tensor(1.0)
            subsampling_weight = subsampling_weight.to(device)
            batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)

            if mode == "tail-batch":
                head_ids = [list[0] for list in positive_sample]
                relations_ids = [list[1] for list in positive_sample]
                head_embed, _,relation_embed= model(
                    embedding = embedding,
                    protein_ids = head_ids,
                    relation_ids = relations_ids,
                    # protein_seq_inputs = protein_seq_inputs
                )
                head_embed = head_embed.unsqueeze(1)
                relation_embed = relation_embed.unsqueeze(1)

                if positive_sample is not None:
                    tail_ids = [list[2] for list in positive_sample]

                    tail_embed, _,_= model(
                        embedding = embedding,
                        protein_ids = tail_ids,
                        # protein_seq_inputs = protein_seq_inputs
                                           )                    
                    tail_embed = tail_embed.unsqueeze(1)
                    score = model_func[self.args.model]( head = head_embed, 
                                                            relation = relation_embed,
                                                            tail = tail_embed)
                    ke_loss = F.logsigmoid(score).squeeze(dim = 1)
                    pos_ke_loss = (- (subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

                if negative_sample is not None:
                    tail_ids = negative_sample.reshape(-1)

                    tail_embed, _,_= model(protein_ids = tail_ids,embedding = embedding,
                                        #    protein_seq_inputs = protein_seq_inputs
                                           )                       
                    tail_embed = tail_embed.reshape(batch_size, negative_sample_size, -1)                 
                    score = model_func[self.args.model]( head = head_embed, 
                                                            relation = relation_embed,
                                                            tail = tail_embed)
                    ke_loss = F.logsigmoid(-score).squeeze(dim = 1)
                    neg_ke_loss = (- (subsampling_weight * ke_loss.mean(dim = 1)).sum()/subsampling_weight.sum())

            if mode == "head-batch":
                tail_ids = [list[2] for list in positive_sample]
                relations_ids = [list[1] for list in positive_sample]
                tail_embed,_,relation_embed= model(
                    embedding = embedding,
                    protein_ids = tail_ids,
                    relation_ids = relations_ids,
                    # protein_seq_inputs = protein_seq_inputs
                )    
                tail_embed = tail_embed.unsqueeze(1)
                relation_embed = relation_embed.unsqueeze(1)         
                if positive_sample is not None:
                    head_ids = [list[0] for list in positive_sample]
                    head_embed, _,_= model(protein_ids = head_ids,embedding = embedding,
                                        #    protein_seq_inputs = protein_seq_inputs
                                           )                    
                    head_embed = head_embed.unsqueeze(1)
                    score = model_func[self.args.model]( head = head_embed, 
                                                            relation = relation_embed,
                                                            tail = tail_embed,
                                                            mode = "head-batch")
                    ke_loss = F.logsigmoid(score).squeeze(dim = 1)
                    pos_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

                if negative_sample is not None:
                    head_ids = negative_sample.reshape(-1)
                    # tail_embed = self.onto_model.go_embedding[tail_ids].reshape(batch_size, negative_sample_size, -1)  
                    head_embed, _,_= model(protein_ids = head_ids,embedding = embedding,
                                        #    protein_seq_inputs = protein_seq_inputs
                                           )                       
                    head_embed = head_embed.reshape(batch_size, negative_sample_size, -1)             
                    score = model_func[self.args.model]( head = head_embed, 
                                                            relation = relation_embed,
                                                            tail = tail_embed,
                                                            mode = "head-batch")
                    ke_loss = F.logsigmoid(-score).squeeze(dim = 1)
                    neg_ke_loss = (- (subsampling_weight * ke_loss.mean(dim = 1)).sum()/subsampling_weight.sum())
            protein_protein_loss = (pos_ke_loss + neg_ke_loss)/2
            total_loss += protein_protein_loss
            all_loss['protein_protein_loss'] = protein_protein_loss

        if  kgg_kgg_inputs:
            positive_sample, negative_sample, subsampling_weight, mode = kgg_kgg_inputs
            # subsampling_weight = torch.tensor(1.0)
            subsampling_weight = subsampling_weight.to(device)
            batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)

            if mode == "tail-batch":
                head_ids = [list[0] for list in positive_sample]
                relations_ids = [list[1] for list in positive_sample]
                _, head_embed,relation_embed= model(
                    go_ids = head_ids,
                    relation_ids = relations_ids,
                )     
                head_embed = head_embed.unsqueeze(1)    
                relation_embed = relation_embed.unsqueeze(1)       
                if positive_sample is not None:
                    tail_ids = [list[2] for list in positive_sample]
                    _, tail_embed,_= model(go_ids = tail_ids,)                    
                    tail_embed = tail_embed.unsqueeze(1)
                    score = model_func[self.args.model]( head = head_embed, 
                                                            relation = relation_embed,
                                                            tail = tail_embed)
                    ke_loss = F.logsigmoid(score).squeeze(dim = 1)
                    pos_ke_loss = (- (subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

                if negative_sample is not None:
                    tail_ids = negative_sample.reshape(-1)

                    _, tail_embed,_= model(go_ids = tail_ids,)                        
                    tail_embed = tail_embed.reshape(batch_size, negative_sample_size, -1)
                    score = model_func[self.args.model]( head = head_embed, 
                                                            relation = relation_embed,
                                                            tail = tail_embed)
                    ke_loss = F.logsigmoid(-score).squeeze(dim = 1)
                    neg_ke_loss = (- (subsampling_weight * ke_loss.mean(dim = 1)).sum()/subsampling_weight.sum())

            if mode == "head-batch":
                tail_ids = [list[2] for list in positive_sample]
                relation_ids = [list[1] for list in positive_sample]
                _, tail_embed,relation_embed= model(
                    go_ids = tail_ids,
                    relation_ids = relation_ids,
                )  
                tail_embed = tail_embed.unsqueeze(1)
                relation_embed = relation_embed.unsqueeze(1)
                if positive_sample is not None:
                    head_ids = [list[0] for list in positive_sample]

                    _, head_embed,_= model(go_ids = head_ids,)                    
                    head_embed = head_embed.unsqueeze(1)
                    score = model_func[self.args.model]( head = head_embed, 
                                                            relation = relation_embed,
                                                            tail = tail_embed,
                                                            mode = "head-batch")
                    ke_loss = F.logsigmoid(score).squeeze(dim = 1)
                    pos_ke_loss = (- (subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

                if negative_sample is not None:
                    head_ids = negative_sample.reshape(-1)
                    # tail_embed = self.onto_model.go_embedding[tail_ids].reshape(batch_size, negative_sample_size, -1)
  
                    _, head_embed,_= model(go_ids = head_ids)                       
                    head_embed = head_embed.reshape(batch_size, negative_sample_size, -1)             
                    score = model_func[self.args.model]( head = head_embed, 
                                                            relation = relation_embed,
                                                            tail = tail_embed,
                                                            mode = "head-batch")
                    ke_loss = F.logsigmoid(-score).squeeze(dim = 1)
                    neg_ke_loss = (- (subsampling_weight * ke_loss.mean(dim = 1)).sum()/subsampling_weight.sum())

            go_go_loss = (pos_ke_loss + neg_ke_loss)/2
            # total_loss += go_go_loss
            all_loss['go_go_loss'] = go_go_loss

        return total_loss, all_loss




    def TransE(self, head, relation, tail, mode = None):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode = None):
        if mode == 'head-batch':
            score = head * (relation * tail) 
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode = None):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode = None):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score
       
    def pRotatE(self, head, relation, tail, mode = None):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    def TransE(self, head, relation, tail, mode = None):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode = None):
        if mode == 'head-batch':
            score = head * (relation * tail) 
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score
