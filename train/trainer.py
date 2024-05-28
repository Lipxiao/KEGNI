import os
import collections
from typing import Optional, Tuple, Union, Dict, Any, List

import torch
import torch.nn as nn
from transformers import Trainer, PreTrainedModel

from loss import KGEloss, MAEloss

import torch.optim as optim
import numpy as np
import pandas as pd


class Trainer(Trainer):

    def __init__(
        self,
        model: Union[nn.Module, PreTrainedModel],
        # args,
        sc_dataset_iter=None,
        kgg_kgg_iter=None,
        scg_kgg_iter=None,
        scg_scg_iter=None,
        kgg_scg_iter=None,
        logger=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            # args=args,
        )
        self.sc_dataset_iter = sc_dataset_iter
        self.kgg_kgg_iter = kgg_kgg_iter
        self.scg_scg_iter = scg_scg_iter
        self.scg_kgg_iter = scg_kgg_iter
        self.kgg_scg_iter = kgg_scg_iter

        # self.args = parser_args()
        self.ke_loss_fn = KGEloss(args=self.args,
                                  # kge_args = self.kge_args
                                  )
        self.mlm_loss_fn = MAEloss(args=self.args)
        self.logger = logger

    def recon(self, z):
        # batch_size = 100  # 每个批次的大小
        # pred = torch.matmul(z,z.t())

        norm = self.args.norm

        def cosine_similarity(tensor):
            # 计算每个向量的 L5 范数
            norms = torch.norm(tensor, p=norm, dim=1)
            # 将每个向量除以其范数
            normalized_tensor = tensor / norms.unsqueeze(1)
            # 计算点积
            dot_product_matrix = torch.mm(normalized_tensor, normalized_tensor.t())
            return dot_product_matrix
        z = torch.tanh(z)
        pred = cosine_similarity(z)

        return pred

    def train(
        self,
        # resume_from_checkpoint: Optional[Union[str, bool]] = None,
        # trial: Union["optuna.Trial", Dict[str, Any]] = None,
        # ignore_keys_for_eval: Optional[List[str]] = None,
        # **kwargs,
    ):
        """
        Rewrite '~transformers.Trainer.train'
        """
        logger = self.logger
        logger.info("args: %s", self.args)
        if self.args.device < 0:
            self.device = "cpu"
        else:
            self.device = f"cuda:{self.args.device}" if torch.cuda.is_available() else "cpu"

        max_steps = self.args.max_steps
        model = self.model
        sc_dataset_iter = self.sc_dataset_iter
        kgg_kgg_iter, scg_kgg_iter, scg_scg_iter, kgg_scg_iter = self.kgg_kgg_iter, self.scg_kgg_iter, self.scg_scg_iter, self.kgg_scg_iter

        self.optimizer = optim.Adam([
            {'params': model.protein_lm.parameters(), 'lr': self.args.mae_lr, 'weight_decay': self.args.mae_weight_decay},
            {'params': model.onto_model.parameters(), 'lr': self.args.kge_lr}
        ])

        def lm_scheduler(step): return (1 + np.cos((step) * np.pi / max_steps)) * 0.5
        def ke_scheduler(step): return ((max_steps - step) / float(max(1, max_steps)))
        # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
        # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
        # self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=[lm_scheduler,ke_scheduler])

        tr_loss = torch.tensor(0.0).to(self.device)
        # self.loss_recorder = []
        # self._total_loss_scalar = 0.0
        # self._globalstep_last_logged = self.state.global_step
        self.optimizer.zero_grad()

        mlm_loss = []
        protein_go_loss = []
        protein_protein_loss = []
        go_protein_loss = []
        go_go_loss = []
        Eprec_values = {}

        if self.args.checkpoint:
            checkpoint = torch.load(self.args.checkpoint)  # 加载断点

            model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
            model.to(self.device)
            self.optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            start_step = checkpoint['step']  # 设置开始的epoch
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            for _ in range(start_step+1):
                next(sc_dataset_iter)[1].to(self.device)
            for _ in range(start_step+1):
                next(kgg_kgg_iter)
            for _ in range(start_step+1):
                next(scg_kgg_iter)
            for _ in range(start_step+1):
                next(scg_scg_iter)
        else:
            start_step = -1
        basename = os.path.splitext(os.path.basename(logger.handlers[0].baseFilename))[0]
        train_iterator = range(
            start_step+1, max_steps
        )
        for step in train_iterator:
            sc_dataset_inputs = None
            kgg_kgg_inputs = None
            scg_kgg_inputs = None
            scg_scg_inputs = None
            kgg_scg_inputs = None

            if sc_dataset_iter:
                sc_dataset_inputs = next(sc_dataset_iter)[1].to(self.device)
            if kgg_kgg_iter:
                kgg_kgg_inputs = next(kgg_kgg_iter)
            if scg_kgg_iter:
                scg_kgg_inputs = next(scg_kgg_iter)
            if scg_scg_iter:
                scg_scg_inputs = next(scg_scg_iter)
            if kgg_scg_iter:
                kgg_scg_inputs = next(kgg_scg_iter)

            loss, all_loss = self.training_step(model, sc_dataset_inputs=sc_dataset_inputs,
                                                kgg_kgg_inputs=kgg_kgg_inputs,
                                                scg_kgg_inputs=scg_kgg_inputs,
                                                kgg_scg_inputs=kgg_scg_inputs,
                                                scg_scg_inputs=scg_scg_inputs)
            mlm_loss.append(all_loss['mlm_loss'])
            protein_go_loss.append(all_loss['protein_go_loss'])
            go_protein_loss.append(all_loss['go_protein_loss'])
            protein_protein_loss.append(all_loss['protein_protein_loss'])
            go_go_loss.append(all_loss['go_go_loss'])
            tr_loss += loss.item()
            if (step + 1) % 1 == 0:
                # print("setp:", '%04d' % (step + 1), "train_loss=", "{:.5f}".format(loss.item()))
                logger.info("step: %04d train_loss= %.5f", step + 1, loss)
            if (step + 1) % 2 == 0:
                self.optimizer.step()
                # self.lr_scheduler.step()
                self.optimizer.zero_grad()
                logger.info("step: %04d two_train_loss= %.5f", step + 1, tr_loss)
                # print("setp:", '%04d' % (step + 1), "two_train_loss=", "{:.5f}".format(tr_loss.item()))
                tr_loss = torch.tensor(0.0)

            # if (step + 1) % 2 == 0:
            #     with torch.no_grad():
            #         model.eval()
            #         z = model.protein_lm.embed(protein_seq_inputs,protein_seq_inputs.ndata['feat'])
            #         # rep = model.protein_lm.encoder_to_decoder(z)
            #         # recon = model.protein_lm.decoder(protein_seq_inputs, rep)

            #         # pred = self.recon(z)
            #         # # pred = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)

            #         # if self.args.device >= 0:
            #         #     pred = pred.cpu()
            #         # pred = pd.DataFrame(pred.numpy(),index= list(model.protein2id.keys()),columns =list(model.protein2id.keys()))
            #         # pred_new = pred.copy()
            #         # pred_new.loc[:,"names"] = pred_new.index
            #         # pred_new = pred_new.melt(id_vars=['names'])
            #         # pred_new = pred_new[pred_new.iloc[:,0] != pred_new.iloc[:,1]]
            #         # pred_new.columns = ["Gene1","Gene2","EdgeWeight"]
            #         # pred_new.sort_values("EdgeWeight",inplace=True,ascending=False)

            #         # name = 'mDC'
            #         # STRING = '/media/disk/project/crosstalk/BEELINE/Beeline/inputs/scRNA/' + name + '/'+''.join([name,'-STRING-network.csv'])
            #         # ChIP = '/media/disk/project/crosstalk/BEELINE/Beeline/inputs/scRNA/' + name + '/'+''.join([name,'-ChIP-network.csv'])
            #         # NonSpe = '/media/disk/project/crosstalk/BEELINE/Beeline/inputs/scRNA/' + name + '/'+''.join([name,'-NonSpe-network.csv'])
            #         # from src.utility import EarlyPrec
            #         # for i in [STRING,NonSpe,ChIP]:
            #         #     predEdgeDF =pred_new.copy()
            #         #     predEdgeDF.columns = ['Gene1','Gene2','EdgeWeight']
            #         #     predEdgeDF['Gene1'] = predEdgeDF['Gene1'].str.upper()
            #         #     predEdgeDF['Gene2'] = predEdgeDF['Gene2'].str.upper()
            #         #     trueEdgesDF = pd.read_csv(i, sep = ',',header = 0, index_col = None)
            #         #     unique_genes = pd.concat([predEdgeDF['Gene2'], predEdgeDF['Gene1']]).unique()
            #         #     netDF = trueEdgesDF.iloc[:, :2].copy()
            #         #     netDF.columns = ['Gene1','Gene2']
            #         #     netDF['Gene1'] = netDF['Gene1'].str.upper()
            #         #     netDF['Gene2'] = netDF['Gene2'].str.upper()
            #         #     netDF = netDF[(netDF.Gene1.isin(unique_genes)) & (netDF.Gene2.isin(unique_genes))]
            #         #     # Remove self-loops.
            #         #     netDF = netDF[netDF.Gene1 != netDF.Gene2]
            #         #     # Remove duplicates (there are some repeated lines in the ground-truth networks!!!).
            #         #     netDF.drop_duplicates(keep = 'first', inplace=True)
            #         #     unique_gene1 = netDF['Gene1'].unique()
            #         #     unique_gene2_combined = pd.concat([netDF['Gene2'], netDF['Gene1']]).unique()
            #         #     predEdgeDF = predEdgeDF[
            #         #         predEdgeDF['Gene1'].isin(unique_gene1) &
            #         #         predEdgeDF['Gene2'].isin(unique_gene2_combined)
            #         #     ]
            #         #     print(EarlyPrec(netDF,predEdgeDF))

            #         # # print('##Beeline',':','\n',self.EarlyPrec(pred_new,self.args.name))
            #         # print('--------------------------------------')

            #     if self.args.eval:
            #         pass
            #         # STRING = '/media/disk/project/crosstalk/BEELINE/Beeline/inputs/scRNA/' + name + '/'+''.join([name,'-STRING-network.csv'])
            #         # ChIP = '/media/disk/project/crosstalk/BEELINE/Beeline/inputs/scRNA/' + name + '/'+''.join([name,'-ChIP-network.csv'])
            #         # NonSpe = '/media/disk/project/crosstalk/BEELINE/Beeline/inputs/scRNA/' + name + '/'+''.join([name,'-NonSpe-network.csv'])
            #         # Eprec = {}
            #         # trueEdges = STRING
            #         # trueEdgesDF = pd.read_csv(trueEdges, sep = ',',header = 0, index_col = None)

            #         # Eprec = EarlyPrec(trueEdgesDF,pred_new)
            #         # logger.info('##Beeline:\n%s', pd.DataFrame(Eprec,index = [0]))
            #         # for key, value in Eprec.items():
            #         #     if key not in Eprec_values:
            #         #         Eprec_values[key] = []
            #         #     Eprec_values[key].append(value)

            # if (step + 1) % 100 == 0:
            #     checkpoint = {
            #             "net": model.state_dict(),
            #             'optimizer': self.optimizer.state_dict(),
            #             "step": step,
            #             'lr_scheduler': self.lr_scheduler.state_dict()
            #         }
            #     if not os.path.exists('./model_parameter/'):  # 判断文件夹是否存在
            #         os.makedirs('./model_parameter/')
            #     file_path = './model_parameter/ckpt_{}_{}.pth'.format(basename, step+1)
            #     torch.save(checkpoint, file_path)

            # if (step+1) % 30000 == 0:
            #     self._save_checkpoint()

        with torch.no_grad():
            model.eval()
            z = model.protein_lm.embed(sc_dataset_inputs, sc_dataset_inputs.ndata['feat'])

            rep = model.protein_lm.encoder_to_decoder(z)
            recon = model.protein_lm.decoder(sc_dataset_inputs, rep)

            pred = self.recon(z)
            if self.args.device >= 0:
                pred = pred.cpu()
            pred = pd.DataFrame(pred.numpy(), index=list(model.protein2id.keys()), columns=list(model.protein2id.keys()))
            pred_new = pred.copy()
            pred_new.loc[:, "names"] = pred_new.index
            pred_new = pred_new.melt(id_vars=['names'])
            pred_new = pred_new[pred_new.iloc[:, 0] != pred_new.iloc[:, 1]]
            pred_new.columns = ["Gene1", "Gene2", "EdgeWeight"]
            pred_new.sort_values("EdgeWeight", inplace=True, ascending=False)

        if not os.path.exists('./outputs/'):
            os.makedirs('./outputs/')
        pd.DataFrame(z.detach().cpu().numpy(), index=list(model.protein2id.keys())).to_csv('./outputs/'+basename+'-embedding.csv')
        pd.DataFrame(model.onto_model.relation_embedding.detach().cpu().numpy(), index=model.relation2id.keys()).to_csv('./outputs/'+basename+'-relation.csv')
        pd.DataFrame(model.onto_model.go_embedding.detach().cpu().numpy(), index=model.entity2id.keys()).to_csv('./outputs/'+basename+'-goembedding.csv')

        # pd.DataFrame(recon.cpu().numpy(),index= list(model.protein2id.keys())).to_csv('./outputs/'+basename+'-recon.csv')

        outname = os.path.splitext(os.path.basename(self.args.input))[0]
        pred_new.to_csv('/media/disk/project/e2e_model/outputs/beeline/' + ''.join([outname, '.txt']), sep='\t', index=False)

        logger.info("Training completed.")

        # from matplotlib.backends.backend_pdf import PdfPages
        # plt_name = './outputs' + basename
        # import pickle as pkl
        # with open(plt_name+'.pkl', 'wb') as f:
        #     pkl.dump((loss,protein_go_loss, protein_protein_loss, mlm_loss,go_go_losszeng,Eprec_values,Eprec), f)

        # with PdfPages(plt_name + '.pdf') as pdf:
        #     plt.figure(figsize =( 8,10))
        #     plt.plot(train_iterator, protein_go_loss,  color='green',label='protein_go_loss')
        #     plt.plot(train_iterator, protein_protein_loss, color='red', label='protein_protein_loss')
        #     # 调整 y 轴刻度间隔
        #     plt.yticks(list(set([round(x, 1) for x in protein_go_loss]) | set([round(x, 1) for x in protein_protein_loss])))
        #     plt.xlabel('Steps')
        #     plt.ylabel('Loss')
        #     plt.legend()
        #     pdf.savefig()
        #     plt.close()

        #     plt.figure()
        #     plt.plot(train_iterator, mlm_loss,  color='blue',label='mlm_loss')
        #     plt.plot(train_iterator, go_go_loss, color='black', label='go_go_loss')
        #     plt.yticks(list(set([round(x, 1) for x in mlm_loss]) | set([round(x, 1) for x in go_go_loss])))
        #     plt.xlabel('Steps')
        #     plt.ylabel('Loss')
        #     plt.legend()
        #     pdf.savefig()
        #     plt.close()

        #     plt.figure()
        #     colors = ['red', 'green', 'blue']
        #     for idx, key in enumerate(Eprec_values.keys()):
        #         eval_steps = [x * (max_steps/len(Eprec_values[key])) for x in range(1,len(Eprec_values[key])+1)]
        #         plt.plot(eval_steps,Eprec_values[key], label=key, color=colors[idx])

        #     # plt.yticks(list(set([round(x, 2) for x in mlm_loss])&set([round(x, 2) for x in go_go_loss])))
        #     plt.legend()
        #     plt.title('Eval Results')
        #     plt.xlabel('Steps')
        #     plt.ylabel('Epr')
        #     pdf.savefig()
        #     plt.close()

    def training_step(
        self,
        model: nn.Module,
        sc_dataset_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        kgg_kgg_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        scg_kgg_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        kgg_scg_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        scg_scg_inputs: Dict[str, Union[torch.Tensor, Any]] = None
    ) -> torch.Tensor:

        model.train()
        loss, all_loss = self.compute_loss(model,
                                           sc_dataset_inputs=sc_dataset_inputs,
                                           kgg_kgg_inputs=kgg_kgg_inputs,
                                           scg_kgg_inputs=scg_kgg_inputs,
                                           kgg_scg_inputs=kgg_scg_inputs,
                                           scg_scg_inputs=scg_scg_inputs)
        loss.backward()
        return loss, all_loss

    def compute_loss(
        self,
        model,
        sc_dataset_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        kgg_kgg_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        scg_kgg_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        kgg_scg_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        scg_scg_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
    ):

        total_loss = torch.tensor(0.0).to(self.device)

        all_loss = collections.defaultdict(float)

        if sc_dataset_inputs:
            mlm_loss, z = self.mlm_loss_fn(model=model, protein_seq_inputs=sc_dataset_inputs)
            total_loss += mlm_loss
            # all_loss['mlm'] = mlm_loss.item()

        ke_total_loss, ke_all_loss = self.ke_loss_fn(model=model, embedding=z,
                                                     scg_scg_inputs=scg_scg_inputs,
                                                     scg_kgg_inputs=scg_kgg_inputs,
                                                     kgg_scg_inputs=kgg_scg_inputs,
                                                     kgg_kgg_inputs=kgg_kgg_inputs)
        ke_all_loss.setdefault("protein_go_loss", torch.tensor(0))
        ke_all_loss.setdefault("go_protein_loss", torch.tensor(0))
        ke_all_loss.setdefault("protein_protein_loss", torch.tensor(0))
        ke_all_loss.setdefault("go_go_loss", torch.tensor(0))

        self.logger.info("mlm_loss: %.5f, protein_go_loss: %.5f, go_protein_loss:%.5f,protein_protein_loss: %.5f, go_go_loss: %.5f",
                         mlm_loss.item(), ke_all_loss['protein_go_loss'].item(), ke_all_loss['go_protein_loss'].item(), ke_all_loss['protein_protein_loss'].item(), ke_all_loss['go_go_loss'].item())

        # print("protein_protein_loss:",ke_all_loss['protein_protein_loss'].item())
        # print("go_go_loss:",ke_all_loss['go_go_loss'].item())
        all_loss['protein_go_loss'] = ke_all_loss['protein_go_loss'].item()
        all_loss['go_protein_loss'] = ke_all_loss['go_protein_loss'].item()
        all_loss['protein_protein_loss'] = ke_all_loss['protein_protein_loss'].item()
        all_loss['go_go_loss'] = ke_all_loss['go_go_loss'].item()
        all_loss['mlm_loss'] = mlm_loss.item()
        total_loss += ke_total_loss
        # all_loss['ke'] = ke_all_loss

        return total_loss, all_loss
