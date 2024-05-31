#!/usr/bin/python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dataset import KGEDataset
from torch.utils.data import DataLoader
import datetime
import pandas as pd

current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")

log_filename = f"my_log_{formatted_time}.log"


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data


class KGEdataloader():
    def __init__(self, **kwargs):
        sc_dataset = kwargs.get("sc_dataset")
        self.negative_sample_size = kwargs.get("negative_sample_size")
        self.batch_size = kwargs.get("batch_size")
        kge_file_path = kwargs.get("data_path")
        kge_data = pd.read_csv(kge_file_path, sep='\t', header=None)
        self.data_process(kge_data, sc_dataset)

    def data_process(self, kge_data, sc_dataset):
        kge_data[0] = kge_data[0].str.upper()
        kge_data[2] = kge_data[2].str.upper()
        kgg = (set(kge_data[0].tolist() + kge_data[2].tolist()))-set(sc_dataset.node2id.keys())

        self.kgg2id = {string: index for index, string in enumerate(sorted(list(kgg)))}
        self.relation2id = {string: index for index, string in enumerate(sorted(list(set(kge_data[1]))))}
        self.scg2id = sc_dataset.node2id
        self.kgg_kgg_triples = []
        self.scg_scg_triples = []
        self.scg_kgg_triples = []
        self.kgg_scg_triples = []

        for index, row in kge_data.iterrows():
            h, r, t = row[0], row[1], row[2]
            if ((h in self.kgg2id) & (t in self.kgg2id)):
                self.kgg_kgg_triples.append((self.kgg2id[h], self.relation2id[r], self.kgg2id[t]))
            elif (h in self.scg2id) & (t in self.scg2id):
                self.scg_scg_triples.append((self.scg2id[h], self.relation2id[r], self.scg2id[t]))
            elif ((h in self.scg2id) & (t in self.kgg2id)):
                self.scg_kgg_triples.append((self.scg2id[h], self.relation2id[r], self.kgg2id[t]))
            elif ((h in self.kgg2id) & (t in self.scg2id)):
                self.kgg_scg_triples.append((self.kgg2id[h], self.relation2id[r], self.scg2id[t]))

    def kgg_kgg_dataloader(self):
        kgg_kgg_dataloader_head = DataLoader(
            KGEDataset(self.kgg_kgg_triples,  negative_sample_size=self.negative_sample_size,
                       mode='head-batch', nentity=len(self.kgg2id)),
            batch_size=int(self.batch_size),
            shuffle=True,
            collate_fn=KGEDataset.collate_fn
        )
        kgg_kgg_dataloader_tail = DataLoader(
            KGEDataset(self.kgg_kgg_triples,  negative_sample_size=self.negative_sample_size,
                       mode='tail-batch', nentity=len(self.kgg2id)),
            batch_size=int(self.batch_size),
            shuffle=True,
            collate_fn=KGEDataset.collate_fn
        )
        kgg_kgg_iter = BidirectionalOneShotIterator(kgg_kgg_dataloader_head, kgg_kgg_dataloader_tail)
        return kgg_kgg_iter

    def scg_scg_dataloader(self):
        scg_scg_dataloader_head = DataLoader(
            KGEDataset(self.scg_scg_triples,  negative_sample_size=self.negative_sample_size,
                       mode='head-batch', nentity=len(self.scg2id)),
            batch_size=int(self.batch_size),
            shuffle=True,
            collate_fn=KGEDataset.collate_fn
        )
        scg_scg_dataloader_tail = DataLoader(
            KGEDataset(self.scg_scg_triples,  negative_sample_size=self.negative_sample_size,
                       mode='tail-batch', nentity=len(self.scg2id)),
            batch_size=int(self.batch_size),
            shuffle=True,
            collate_fn=KGEDataset.collate_fn
        )
        scg_scg_iter = BidirectionalOneShotIterator(scg_scg_dataloader_head, scg_scg_dataloader_tail)
        return scg_scg_iter

    def kgg_scg_dataloader(self):
        kgg_scg_dataloader_head = DataLoader(
            KGEDataset(self.kgg_scg_triples,  negative_sample_size=self.negative_sample_size,
                       mode='head-batch', nentity=len(self.kgg2id)),
            batch_size=int(self.batch_size),
            shuffle=True,
            collate_fn=KGEDataset.collate_fn
        )
        kgg_scg_iter = BidirectionalOneShotIterator.one_shot_iterator(kgg_scg_dataloader_head)
        return kgg_scg_iter

    def scg_kgg_dataloader(self):
        scg_kgg_dataloader_head = DataLoader(
            KGEDataset(self.scg_kgg_triples,  negative_sample_size=self.negative_sample_size,
                       mode='head-batch', nentity=len(self.scg2id)),
            batch_size=int(self.batch_size),
            shuffle=True,
            collate_fn=KGEDataset.collate_fn
        )
        scg_kgg_dataloader_tail = DataLoader(
            KGEDataset(self.scg_kgg_triples,  negative_sample_size=self.negative_sample_size,
                       mode='tail-batch', nentity=len(self.kgg2id)),
            batch_size=int(self.batch_size),
            shuffle=True,
            collate_fn=KGEDataset.collate_fn
        )
        scg_kgg_iter = BidirectionalOneShotIterator(scg_kgg_dataloader_head, scg_kgg_dataloader_tail)
        # scg_kgg_iter = BidirectionalOneShotIterator.one_shot_iterator(scg_kgg_dataloader_tail)
        return scg_kgg_iter
