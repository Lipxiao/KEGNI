#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class KGEmodel(nn.Module):
    def __init__(self,  nscg, nkgg, nrelation, num_hidden, gamma):
        super(KGEmodel, self).__init__()
        self.nrelation = nrelation
        self.num_hidden = num_hidden
        epsilon = 2.0
        self.nscg = nscg
        self.nkgg = nkgg
        self.num_hidden = num_hidden
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + epsilon) / num_hidden]),
            requires_grad=False
        )

        # self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights"""
        self.kgg_embedding = nn.Parameter(torch.zeros(self.nkgg, self.num_hidden))
        nn.init.uniform_(
            tensor=self.kgg_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        # nn.init.xavier_normal_(
        #     tensor=self.kgg_embedding,
        #     gain = self.gamma.item()
        # )
        # nn.init.normal_(
        #     tensor=self.kgg_embedding,
        #     mean=0,
        #     std=1
        # )
        # nn.init.uniform_(
        #     tensor=self.kgg_embedding,
        #     a=-0.1,
        #     b=0.1
        # )
        self.relation_embedding = nn.Parameter(torch.zeros(self.nrelation, self.num_hidden))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # nn.init.uniform_(
        #     tensor=self.relation_embedding,
        #     a=-0.1,
        #     b=0.1
        # )
        # nn.init.xavier_normal_(
        #     tensor=self.relation_embedding,
        #     gain = self.gamma.item()
        # )
        # nn.init.normal_(
        #     tensor=self.relation_embedding,
        #     mean=0,
        #     std=1
        # )

    def forward(self,
                kgg_ids=None,
                relation_ids=None):

        kgg_embedding = None
        relation_embedding = None
        if kgg_ids is not None:
            kgg_embedding = self.kgg_embedding[[int(tensor_id.item()) for tensor_id in kgg_ids]]
        if relation_ids is not None:
            relation_embedding = self.relation_embedding[[int(tensor_id.item()) for tensor_id in relation_ids]]
        return kgg_embedding, relation_embedding
