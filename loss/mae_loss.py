import torch
import torch.nn as nn
import torch.nn.functional as F


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss

# @dataclass
class MAEloss:
    """
    Loss function for MLM.

    Args:
        mlm_lambda: hyper-parameters to control the effect of MLM loss.
    """
    def __init__(self,args):
        self.args = args
        self.linear = nn.Linear(args.num_hidden, args.num_hidden)

    def __call__(
        self,
        model,
        protein_seq_inputs
    ):
        
        if self.args.device < 0:
            device = "cpu"
        else:
            device = f"cuda:{self.args.device}" if torch.cuda.is_available() else "cpu"

        graph = protein_seq_inputs
        x = graph.ndata["feat"]
        protein_lm = model.protein_lm.to(device)
        loss, embed = protein_lm(graph, x)
        # z = embed
        z = self.linear.to(device)(embed)

        return loss ,z
