import os
import collections
from typing import  Union, Dict, Any
import torch
import torch.nn as nn
from transformers import Trainer
from loss import KGEloss, MAEloss
import torch.optim as optim
import numpy as np
import pandas as pd
from eval import MultiEval
from utils.utility import save_ckpt
from datetime import datetime
class Trainer(Trainer):
    def __init__(
        self,
        model: Union[nn.Module],
        args = None,
        sc_dataset_iter=None,
        kgg_kgg_iter=None,
        scg_kgg_iter=None,
        scg_scg_iter=None,
        kgg_scg_iter=None,
        logger=None
    ):
        super().__init__(model=model)
        self.args = args
        self.sc_dataset_iter = sc_dataset_iter
        self.kgg_kgg_iter = kgg_kgg_iter
        self.scg_scg_iter = scg_scg_iter
        self.scg_kgg_iter = scg_kgg_iter
        self.kgg_scg_iter = kgg_scg_iter

        self.kge_loss_fn = KGEloss(args=self.args)
        self.mae_loss_fn = MAEloss(args=self.args)
        self.logger = logger

    def train(self):
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
            {'params': model.mae_model.parameters(), 'lr': self.args.mae_lr, 'weight_decay': self.args.mae_weight_decay},
            {'params': model.kge_model.parameters(), 'lr': self.args.kge_lr}
        ])
        self.optimizer.zero_grad()
        
        def lm_scheduler(step): return (1 + np.cos((step) * np.pi / max_steps)) * 0.5
        def ke_scheduler(step): return ((max_steps - step) / float(max(1, max_steps)))

        tr_loss = torch.tensor(0.0).to(self.device)
        mae_loss = []
        scg_kgg_loss = []
        scg_scg_loss = []
        kgg_scg_loss = []
        kgg_kgg_loss = []
        epr_values = []
        pr_values = []
        roc_values = []
        best_models = {}
        
        if self.args.load_checkpoint:
            checkpoint = torch.load(self.args.load_checkpoint) 
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
            start_step = checkpoint['step']  
            # self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            #TODO 是否需要保存dataloader
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
        for step in range(start_step+1, max_steps):
            sc_dataset_inputs = None
            kgg_kgg_inputs = None
            scg_kgg_inputs = None
            scg_scg_inputs = None
            kgg_scg_inputs = None

            if sc_dataset_iter:
                sc_dataset_inputs = next(sc_dataset_iter)[1].to(self.device)#TODO 是否需要to(device)
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
                                                scg_scg_inputs=scg_scg_inputs,
                                                kgg_scg_inputs=kgg_scg_inputs)
            mae_loss.append(all_loss['mae_loss'])
            scg_kgg_loss.append(all_loss['scg_kgg_loss'])
            kgg_scg_loss.append(all_loss['kgg_scg_loss'])
            scg_scg_loss.append(all_loss['scg_scg_loss'])
            kgg_kgg_loss.append(all_loss['kgg_kgg_loss'])
            tr_loss += loss.item()
            if (step + 1) % 1 == 0:
                logger.info("step: %04d train_loss= %.5f", step + 1, loss)
            if (step + 1) % 2 == 0:
                self.optimizer.step()
                # self.lr_scheduler.step()
                self.optimizer.zero_grad()
                logger.info("step: %04d two_train_loss= %.5f", step + 1, tr_loss)
                tr_loss = torch.tensor(0.0) #TODO 前面tooch.tensor(0.0)是否有必要to(device)
                
            if (self.args.eval) & ((step + 1) % 50 == 0):
                print("evaluating...")
                model.eval()  
                with torch.no_grad():
                    z = model.mae_model.embed(sc_dataset_inputs, sc_dataset_inputs.ndata['feat'])
                    predDF = self.recon(z)
                    
                name =basename.split('_')[0]

                if name == "mESC":
                    gt_paths = {
                        'STRING': '/media/disk/project/crosstalk/BEELINE/Beeline/inputs/scRNA/' + name + '/' + name + '-STRING-network.csv',
                        'NonSpe': '/media/disk/project/crosstalk/BEELINE/Beeline/inputs/scRNA/' + name + '/' + name + '-NonSpe-network.csv',
                        'ChIP': '/media/disk/project/crosstalk/BEELINE/Beeline/inputs/scRNA/' + name + '/' + name + '-ChIP-network.csv',
                        'lofgof': '/media/disk/project/crosstalk/BEELINE/Beeline/inputs/scRNA/' + name + '/' + name + '-lofgof-network.csv'}
                else:
                    gt_paths = {
                        'STRING': '/media/disk/project/crosstalk/BEELINE/Beeline/inputs/scRNA/' + name + '/' + name + '-STRING-network.csv',
                        'NonSpe': '/media/disk/project/crosstalk/BEELINE/Beeline/inputs/scRNA/' + name + '/' + name + '-NonSpe-network.csv',
                        'ChIP': '/media/disk/project/crosstalk/BEELINE/Beeline/inputs/scRNA/' + name + '/' + name + '-ChIP-network.csv'}
                
                for i in gt_paths.keys():
                    dataset = gt_paths[i]
                    trueEdgesDF = pd.read_csv(dataset, sep = ',',header = 0, index_col = None)
                    epr,pr,roc = MultiEval(predDF,trueEdgesDF)
                    epr_values.append(epr)
                    pr_values.append(pr)
                    roc_values.append(roc)
                    if i not in best_models or epr > best_models[i]['epr']:
                        best_models[i] =  {'step':step+1,'epr': epr, 'pr': pr, 'roc': roc, 'model': model}
                        if self.args.save_checkpoint:
                            ckpt_folder = './checkpoints/best/'
                            pt_name = name + '_' + i
                            save_ckpt(step,  best_models[i]['model'], self.optimizer,  pt_name,ckpt_folder)
        if self.args.eval:    
            for key in best_models:
                best_info = best_models[key]
                self.logger.info(
                    "%s: best_epr: %.5f, step: %d",
                    key,
                    best_info['epr'],
                    best_info['step']
                )
                
            df = pd.DataFrame({
                'max_steps': [max_steps],
                **{f"best_epr_{key}": f"{info['epr']:.5f}" for key, info in best_models.items()},
                **{f"step_{key}": info['step'] for key, info in best_models.items()}
            })

            save_path = 'results/' + basename.split('_')[0]+'/'
            current_date = datetime.now().strftime("%Y%m%d%H")
            filename = f"{save_path}{basename.split('_')[0]}_{self.args.n_neighbors}_{self.args.num_hidden}_{self.args.num_heads}_{self.args.num_layers}_{current_date}.csv"
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(os.path.join(filename), index=False)
                                  
        model.eval()
        with torch.no_grad():
            z = model.mae_model.embed(sc_dataset_inputs, sc_dataset_inputs.ndata['feat'])
            predDF = self.recon(z)
            recon_exp = model.mae_model.decoder(sc_dataset_inputs, model.mae_model.encoder_to_decoder(z))

        if not os.path.exists('./outputs/'):
            os.makedirs('./outputs/')
        pd.DataFrame(z.detach().cpu().numpy(), index=list(model.scg2id.keys())).to_csv('./outputs/'+basename+'-scg_embedding.csv')
        pd.DataFrame(model.kge_model.relation_embedding.detach().cpu().numpy(), index=model.relation2id.keys()).to_csv('./outputs/'+basename+'-relation_embedding.csv')
        pd.DataFrame(model.kge_model.kgg_embedding.detach().cpu().numpy(), index=model.kgg2id.keys()).to_csv('./outputs/'+basename+'-kgg_embedding.csv')

        # pd.DataFrame(recon_exp.cpu().numpy(),index= list(model.scg2id.keys())).to_csv('./outputs/'+basename+'-recon.csv')

        outname = os.path.splitext(os.path.basename(self.args.input))[0]
        predDF.to_csv('/media/disk/project/KEGIN/outputs/pred/' + ''.join([outname, '.txt']), sep='\t', index=False)
        if self.args.save_checkpoint:
            ckpt_folder = './checkpoints/'
            save_ckpt(step, model, self.optimizer,  basename,ckpt_folder)
        
        #TODO 正式代码中可以不保存    
        import pickle as pkl
        save_path = 'results/' + basename.split('_')[0]+'/'
        current_date = datetime.now().strftime("%Y%m%d%H")
        filename = f"{save_path}{basename.split('_')[0]}_{self.args.n_neighbors}_{self.args.num_hidden}_{self.args.num_heads}_{self.args.num_layers}_{current_date}_dict.pkl"
        os.makedirs(save_path, exist_ok=True)
        with open(filename, 'wb') as f:
            pkl.dump((loss,mae_loss, kgg_kgg_loss, kgg_scg_loss, scg_scg_loss, scg_kgg_loss,epr_values,pr_values,roc_values), f)
            
        logger.info("Training completed.")


    def training_step(
        self,
        model: nn.Module,
        sc_dataset_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        kgg_kgg_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        scg_kgg_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        kgg_scg_inputs: Dict[str, Union[torch.Tensor, Any]] = None,
        scg_scg_inputs: Dict[str, Union[torch.Tensor, Any]] = None
    ):

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
            mae_loss, z = self.mae_loss_fn(model=model, sc_dataset_inputs=sc_dataset_inputs)
            total_loss += mae_loss

        kge_total_loss, kge_all_loss = self.kge_loss_fn(model=model, embedding=z,
                                                     scg_scg_inputs=scg_scg_inputs,
                                                     scg_kgg_inputs=scg_kgg_inputs,
                                                     kgg_scg_inputs=kgg_scg_inputs,
                                                     kgg_kgg_inputs=kgg_kgg_inputs)
        kge_all_loss.setdefault("scg_kgg_loss", torch.tensor(0))
        kge_all_loss.setdefault("kgg_scg_loss", torch.tensor(0))
        kge_all_loss.setdefault("scg_scg_loss", torch.tensor(0))
        kge_all_loss.setdefault("kgg_kgg_loss", torch.tensor(0))

        self.logger.info("mae_loss: %.5f, scg_kgg_loss: %.5f, kgg_scg_loss:%.5f,scg_scg_loss: %.5f, kgg_kgg_loss: %.5f",
                         mae_loss.item(), kge_all_loss['scg_kgg_loss'].item(), kge_all_loss['kgg_scg_loss'].item(), kge_all_loss['scg_scg_loss'].item(), kge_all_loss['kgg_kgg_loss'].item())

        all_loss['scg_kgg_loss'] = kge_all_loss['scg_kgg_loss'].item()
        all_loss['kgg_scg_loss'] = kge_all_loss['kgg_scg_loss'].item()
        all_loss['scg_scg_loss'] = kge_all_loss['scg_scg_loss'].item()
        all_loss['kgg_kgg_loss'] = kge_all_loss['kgg_kgg_loss'].item()
        all_loss['mae_loss'] = mae_loss.item()
        total_loss += kge_total_loss

        return total_loss, all_loss
    
    def recon(self, z):
        norm = self.args.norm
        def cosine_similarity(tensor):
            norms = torch.norm(tensor, p=norm, dim=1)
            normalized_tensor = tensor / norms.unsqueeze(1)
            dot_product_matrix = torch.mm(normalized_tensor, normalized_tensor.t())
            return dot_product_matrix
        z = torch.tanh(z)
        pred = cosine_similarity(z)
        
        if self.args.device >= 0:
            pred = pred.cpu()
        pred = pd.DataFrame(pred.numpy(), index=list(self.model.scg2id.keys()), columns=list(self.model.scg2id.keys()))
        predDF = pred.copy()
        predDF.loc[:, "names"] = predDF.index
        predDF = predDF.melt(id_vars=['names'])
        predDF = predDF[predDF.iloc[:, 0] != predDF.iloc[:, 1]]
        predDF.columns = ["Gene1", "Gene2", "EdgeWeight"]
        predDF.sort_values("EdgeWeight", inplace=True, ascending=False)
        return predDF