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
# from args import  parser_args

from model.KGE.KGEmodel import KGEmodel
from model.MAE.MAEmodel import MAEmodel
# from kge_model.utils import parse_args

# logger = logging.get_logger(__name__)
logger = logging.getLogger(__name__)

class E2EModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        scg2id = kwargs.get("scg2id")
        relation2id = kwargs.get("relation2id")
        kgg2id = kwargs.get("kgg2id")
        self.device = kwargs.get("device")
        
        num_features = kwargs.get("num_features")
        num_hidden = kwargs.get("num_hidden")
        gamma = kwargs.get("gamma")
        num_layers = kwargs.get("num_layers")
        num_heads = kwargs.get("num_heads")
        num_out_heads = kwargs.get("num_out_heads")
        activation = kwargs.get("activation")
        in_drop = kwargs.get("in_drop")
        attn_drop = kwargs.get("attn_drop")
        negative_slope = kwargs.get("negative_slope")
        residual = kwargs.get("residual")
        encoder_type = kwargs.get("encoder_type")
        decoder_type = kwargs.get("decoder_type")
        mask_rate = kwargs.get("mask_rate")
        norm = kwargs.get("norm")
        loss_fn = kwargs.get("loss_fn")
        drop_edge_rate = kwargs.get("drop_edge_rate")
        replace_rate = kwargs.get("replace_rate")
        alpha_l = kwargs.get("alpha_l")
        concat_hidden = kwargs.get("concat_hidden")
        
        
        
        
        if self.device < 0:
            self.device = "cpu"
        else:
            self.device = f"cuda:{self.device}" if torch.cuda.is_available() else "cpu"
        self.kg_model = KGEmodel(
            nrelation = len(relation2id),
            nscg = len(scg2id),
            nkgg = len(kgg2id),
            num_hidden=num_hidden,
            gamma=gamma)
        
        self.mae_model = MAEmodel(in_dim=num_features,num_hidden=num_hidden,
                                num_layers=num_layers,
                                nhead=num_heads,
                                nhead_out=num_out_heads,
                                activation=activation,
                                feat_drop=in_drop,
                                attn_drop=attn_drop,
                                negative_slope=negative_slope,
                                residual=residual,
                                encoder_type=encoder_type,
                                decoder_type=decoder_type,
                                mask_rate=mask_rate,
                                norm=norm,
                                loss_fn=loss_fn,
                                drop_edge_rate=drop_edge_rate,
                                replace_rate=replace_rate,
                                alpha_l=alpha_l,
                                concat_hidden=concat_hidden)

    def forward(
        self,
        embedding = None,
        scg_ids = None,
        relation_ids = None,
        kgg_ids = None
    ):
        scg_embedding = None
        relation_embedding = None
        kgg_embedding = None

        kg_model =  self.kg_model.to(self.device)

        if scg_ids is not None:
            scg_embedding = embedding[[int(tensor_id.item()) for tensor_id in scg_ids]] 
        kgg_embedding, relation_embedding = kg_model(
            kgg_ids = kgg_ids,
            relation_ids = relation_ids
        )
        return scg_embedding,kgg_embedding,relation_embedding

    # def save_pretrained(
    #     self,
    #     save_directory: os.PathLike,
    #     state_dict: Optional[dict] = None,
    #     save_config: bool = True,
    # ):
    #     protein_save_directory = os.path.join(save_directory, 'protein')
    #     onto_save_directory = os.path.join(save_directory, 'onto')
    #     if self.mae_model:
    #         self.mae_model.save_pretrained(protein_save_directory, save_config=save_config)
    #     self.kg_model.save_pretrained(onto_save_directory, save_config=save_config)

