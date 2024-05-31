
import torch
import torch.nn as nn
import logging

from model.KGE.KGEmodel import KGEmodel
from model.MAE.MAEmodel import MAEmodel

logger = logging.getLogger(__name__)


class KEGNI(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.device = kwargs.get("device")
        self.scg2id = kwargs.get("scg2id")
        self.relation2id = kwargs.get("relation2id")
        self.kgg2id = kwargs.get("kgg2id")
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
        encoder_type = kwargs.get("encoder")
        decoder_type = kwargs.get("decoder")
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
        self.kge_model = KGEmodel(
            nrelation=len(self.relation2id),
            nscg=len(self.scg2id),
            nkgg=len(self.kgg2id),
            num_hidden=num_hidden,
            gamma=gamma)

        self.mae_model = MAEmodel(in_dim=num_features,
                                  num_hidden=num_hidden,
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
        embedding=None,
        scg_ids=None,
        relation_ids=None,
        kgg_ids=None
    ):
        scg_embedding = None
        relation_embedding = None
        kgg_embedding = None

        kge_model = self.kge_model.to(self.device)

        if scg_ids is not None:
            scg_embedding = embedding[[int(tensor_id.item()) for tensor_id in scg_ids]]
        kgg_embedding, relation_embedding = kge_model(
            kgg_ids=kgg_ids,
            relation_ids=relation_ids
        )
        return scg_embedding, kgg_embedding, relation_embedding
