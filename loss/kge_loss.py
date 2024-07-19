import torch
import torch.nn.functional as F
import collections
# from model.models import KEGNI


class KGEloss:
    """
    Loss function for KE.
    """

    def __init__(self, args):
        self.args = args

    def __call__(
        self,
        model,
        embedding=None,
        kgg_kgg_inputs=None,
        scg_kgg_inputs=None,
        kgg_scg_inputs=None,
        scg_scg_inputs=None,
    ):
        
        if self.args.device < 0:
            device = "cpu"
        else:
            # device = f"cuda:{self.args.device}" if torch.cuda.is_available() else "cpu"
            device = "cuda:" + str(self.args.device) if torch.cuda.is_available() else "cpu"
            
        model_func = {
            'TransE': self.TransE,
            'ComplEx': self.ComplEx
        }
        if self.args.model not in model_func:
            raise ValueError('model %s not supported' % self.args.model)

        self.gamma = model.kge_model.gamma
        self.embedding_range = model.kge_model.embedding_range
        total_loss = torch.tensor(0.0).to(device)
        all_loss = collections.defaultdict(float)

        if scg_kgg_inputs:
            positive_sample, negative_sample, subsampling_weight, mode = scg_kgg_inputs
            # subsampling_weight = torch.tensor(1.0)
            subsampling_weight = subsampling_weight.to(device)
            batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
            if mode == "tail-batch":
                head_ids = [list[0] for list in positive_sample]
                relations_ids = [list[1] for list in positive_sample]
                head_embed, _, relation_embed = model(
                    embedding=embedding,
                    scg_ids=head_ids,
                    relation_ids=relations_ids)
                head_embed = head_embed.unsqueeze(1)
                relation_embed = relation_embed.unsqueeze(1)
                if positive_sample is not None:
                    tail_ids = [list[2] for list in positive_sample]
                    _, tail_embed, _ = model(
                        kgg_ids=tail_ids
                    )
                    tail_embed = tail_embed.unsqueeze(1)
                    score = model_func[self.args.model](head=head_embed,
                                                        relation=relation_embed,
                                                        tail=tail_embed,
                                                        mode="tail-batch")
                    ke_loss = F.logsigmoid(score).squeeze(dim=1)
                    pos_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

                if negative_sample is not None:
                    tail_ids = negative_sample.reshape(-1)
                    _,  tail_embed, _ = model(
                        kgg_ids=tail_ids
                    )
                    tail_embed = tail_embed.reshape(batch_size, negative_sample_size, -1)
                    score = model_func[self.args.model](head=head_embed,
                                                        relation=relation_embed,
                                                        tail=tail_embed,
                                                        mode="tail-batch")
                    ke_loss = F.logsigmoid(-score).mean(dim=1)
                    neg_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())
            if mode == "head-batch":
                tail_ids = [list[2] for list in positive_sample]
                relations_ids = [list[1] for list in positive_sample]
                _, tail_embed, relation_embed = model(
                    kgg_ids=tail_ids,
                    relation_ids=relations_ids)
                tail_embed = tail_embed.unsqueeze(1)
                relation_embed = relation_embed.unsqueeze(1)
                if positive_sample is not None:
                    head_ids = [list[0] for list in positive_sample]
                    head_embed, _, _ = model(
                        embedding=embedding,
                        scg_ids=head_ids)
                    head_embed = head_embed.unsqueeze(1)
                    score = model_func[self.args.model](head=head_embed,
                                                        relation=relation_embed,
                                                        tail=tail_embed,
                                                        mode="head-batch")
                    ke_loss = F.logsigmoid(score).squeeze(dim=1)
                    pos_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

                if negative_sample is not None:
                    head_ids = negative_sample.reshape(-1)
                    head_embed, _, _ = model(
                        embedding=embedding,
                        scg_ids=head_ids,
                    )
                    head_embed = head_embed.reshape(batch_size, negative_sample_size, -1)
                    score = model_func[self.args.model](head=head_embed,
                                                        relation=relation_embed,
                                                        tail=tail_embed,
                                                        mode="head-batch")
                    ke_loss = F.logsigmoid(-score).mean(dim=1)
                    neg_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())
            scg_kgg_loss = (pos_ke_loss + neg_ke_loss)/2
            total_loss += scg_kgg_loss
            all_loss['scg_kgg_loss'] = scg_kgg_loss


        if scg_scg_inputs:
            positive_sample, negative_sample, subsampling_weight, mode = scg_scg_inputs
            # subsampling_weight = torch.tensor(1.0)
            subsampling_weight = subsampling_weight.to(device)
            batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
            if mode == "tail-batch":
                head_ids = [list[0] for list in positive_sample]
                relations_ids = [list[1] for list in positive_sample]
                head_embed, _, relation_embed = model(
                    embedding=embedding,
                    scg_ids=head_ids,
                    relation_ids=relations_ids)
                head_embed = head_embed.unsqueeze(1)
                relation_embed = relation_embed.unsqueeze(1)
                if positive_sample is not None:
                    tail_ids = [list[2] for list in positive_sample]
                    tail_embed, _, _ = model(
                        embedding=embedding,
                        scg_ids=tail_ids)
                    tail_embed = tail_embed.unsqueeze(1)
                    score = model_func[self.args.model](head=head_embed,
                                                        relation=relation_embed,
                                                        tail=tail_embed,
                                                        mode="tail-batch")
                    ke_loss = F.logsigmoid(score).squeeze(dim=1)
                    pos_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

                if negative_sample is not None:
                    tail_ids = negative_sample.reshape(-1)

                    tail_embed, _, _ = model(scg_ids=tail_ids, embedding=embedding)
                    tail_embed = tail_embed.reshape(batch_size, negative_sample_size, -1)
                    score = model_func[self.args.model](head=head_embed,
                                                        relation=relation_embed,
                                                        tail=tail_embed,
                                                        mode="tail-batch")
                    ke_loss = F.logsigmoid(-score).mean(dim=1)
                    neg_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

            if mode == "head-batch":
                tail_ids = [list[2] for list in positive_sample]
                relations_ids = [list[1] for list in positive_sample]
                tail_embed, _, relation_embed = model(
                    embedding=embedding,
                    scg_ids=tail_ids,
                    relation_ids=relations_ids)
                tail_embed = tail_embed.unsqueeze(1)
                relation_embed = relation_embed.unsqueeze(1)
                if positive_sample is not None:
                    head_ids = [list[0] for list in positive_sample]
                    head_embed, _, _ = model(scg_ids=head_ids, embedding=embedding)
                    head_embed = head_embed.unsqueeze(1)
                    score = model_func[self.args.model](head=head_embed,
                                                        relation=relation_embed,
                                                        tail=tail_embed,
                                                        mode="head-batch")
                    ke_loss = F.logsigmoid(score).squeeze(dim=1)
                    pos_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

                if negative_sample is not None:
                    head_ids = negative_sample.reshape(-1)
                    head_embed, _, _ = model(scg_ids=head_ids, embedding=embedding)
                    head_embed = head_embed.reshape(batch_size, negative_sample_size, -1)
                    score = model_func[self.args.model](head=head_embed,
                                                        relation=relation_embed,
                                                        tail=tail_embed,
                                                        mode="head-batch")
                    ke_loss = F.logsigmoid(-score).mean(dim=1)
                    neg_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())
            scg_scg_loss = (pos_ke_loss + neg_ke_loss)/2
            total_loss += scg_scg_loss
            all_loss['scg_scg_loss'] = scg_scg_loss

        if kgg_scg_inputs:
            positive_sample, negative_sample, subsampling_weight, mode = kgg_scg_inputs
            subsampling_weight = subsampling_weight.to(device)
            batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)
            if mode == "tail-batch":
                head_ids = [list[0] for list in positive_sample]
                relations_ids = [list[1] for list in positive_sample]
                _, head_embed, relation_embed = model(
                    kgg_ids=head_ids,
                    relation_ids=relations_ids,
                )
                head_embed = head_embed.unsqueeze(1)
                relation_embed = relation_embed.unsqueeze(1)
                if positive_sample is not None:
                    tail_ids = [list[2] for list in positive_sample]
                    tail_embed, _, _ = model(
                        embedding=embedding,
                        scg_ids=tail_ids)
                    tail_embed = tail_embed.unsqueeze(1)
                    score = model_func[self.args.model](head=head_embed,
                                                        relation=relation_embed,
                                                        tail=tail_embed,
                                                        mode="tail-batch")
                    ke_loss = F.logsigmoid(score).squeeze(dim=1)
                    pos_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

                if negative_sample is not None:
                    tail_ids = negative_sample.reshape(-1)
                    tail_embed, _, _ = model(
                        embedding=embedding,
                        scg_ids=tail_ids)
                    tail_embed = tail_embed.reshape(batch_size, negative_sample_size, -1)
                    score = model_func[self.args.model](head=head_embed,
                                                        relation=relation_embed,
                                                        tail=tail_embed,
                                                        mode="tail-batch")
                    ke_loss = F.logsigmoid(-score).mean(dim=1)
                    neg_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

            if mode == "head-batch":
                tail_ids = [list[2] for list in positive_sample]
                relations_ids = [list[1] for list in positive_sample]
                tail_embed, _, relation_embed = model(
                    embedding=embedding,
                    scg_ids=tail_ids,
                    relation_ids=relations_ids)
                tail_embed = tail_embed.unsqueeze(1)
                relation_embed = relation_embed.unsqueeze(1)
                if positive_sample is not None:
                    head_ids = [list[0] for list in positive_sample]
                    _, head_embed, _ = model(kgg_ids=head_ids)
                    head_embed = head_embed.unsqueeze(1)
                    score = model_func[self.args.model](head=head_embed,
                                                        relation=relation_embed,
                                                        tail=tail_embed,
                                                        mode="head-batch")
                    ke_loss = F.logsigmoid(score).squeeze(dim=1)
                    pos_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

                if negative_sample is not None:
                    head_ids = negative_sample.reshape(-1)
                    _, head_embed, _ = model(kgg_ids=head_ids)
                    head_embed = head_embed.reshape(batch_size, negative_sample_size, -1)
                    score = model_func[self.args.model](head=head_embed,
                                                        relation=relation_embed,
                                                        tail=tail_embed,
                                                        mode="head-batch")
                    ke_loss = F.logsigmoid(-score).mean(dim=1) 
                    neg_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())
            kgg_scg_loss = (pos_ke_loss + neg_ke_loss)/2
            # total_loss += kgg_scg_loss
            all_loss['kgg_scg_loss'] = kgg_scg_loss


        if kgg_kgg_inputs:
            positive_sample, negative_sample, subsampling_weight, mode = kgg_kgg_inputs
            # subsampling_weight = torch.tensor(1.0)
            subsampling_weight = subsampling_weight.to(device)
            batch_size, negative_sample_size = negative_sample.size(0), negative_sample.size(1)

            if mode == "tail-batch":
                head_ids = [list[0] for list in positive_sample]
                relations_ids = [list[1] for list in positive_sample]
                _, head_embed, relation_embed = model(
                    kgg_ids=head_ids,
                    relation_ids=relations_ids,
                )
                head_embed = head_embed.unsqueeze(1)
                relation_embed = relation_embed.unsqueeze(1)
                if positive_sample is not None:
                    tail_ids = [list[2] for list in positive_sample]
                    _, tail_embed, _ = model(kgg_ids=tail_ids,)
                    tail_embed = tail_embed.unsqueeze(1)
                    score = model_func[self.args.model](head=head_embed,
                                                        relation=relation_embed,
                                                        tail=tail_embed)
                    ke_loss = F.logsigmoid(score).squeeze(dim=1)
                    pos_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

                if negative_sample is not None:
                    tail_ids = negative_sample.reshape(-1)

                    _, tail_embed, _ = model(kgg_ids=tail_ids,)
                    tail_embed = tail_embed.reshape(batch_size, negative_sample_size, -1)
                    score = model_func[self.args.model](head=head_embed,
                                                        relation=relation_embed,
                                                        tail=tail_embed)
                    ke_loss = F.logsigmoid(-score).mean(dim=1)
                    neg_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

            if mode == "head-batch":
                tail_ids = [list[2] for list in positive_sample]
                relation_ids = [list[1] for list in positive_sample]
                _, tail_embed, relation_embed = model(
                    kgg_ids=tail_ids,
                    relation_ids=relation_ids,
                )
                tail_embed = tail_embed.unsqueeze(1)
                relation_embed = relation_embed.unsqueeze(1)
                if positive_sample is not None:
                    head_ids = [list[0] for list in positive_sample]

                    _, head_embed, _ = model(kgg_ids=head_ids,)
                    head_embed = head_embed.unsqueeze(1)
                    score = model_func[self.args.model](head=head_embed,
                                                        relation=relation_embed,
                                                        tail=tail_embed,
                                                        mode="head-batch")
                    ke_loss = F.logsigmoid(score).squeeze(dim=1)
                    pos_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

                if negative_sample is not None:
                    head_ids = negative_sample.reshape(-1)

                    _, head_embed, _ = model(kgg_ids=head_ids)
                    head_embed = head_embed.reshape(batch_size, negative_sample_size, -1)
                    score = model_func[self.args.model](head=head_embed,
                                                        relation=relation_embed,
                                                        tail=tail_embed,
                                                        mode="head-batch")
                    ke_loss = F.logsigmoid(-score).mean(dim=1)
                    neg_ke_loss = (-(subsampling_weight * ke_loss).sum()/subsampling_weight.sum())

            kgg_kgg_loss = (pos_ke_loss + neg_ke_loss)/2
            # total_loss += kgg_kgg_loss
            all_loss['kgg_kgg_loss'] = kgg_kgg_loss

        return total_loss, all_loss

    def TransE(self, head, relation, tail, mode=None):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode=None):
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

        score = score.sum(dim=2)
        return score