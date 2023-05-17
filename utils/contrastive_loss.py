# Import libraries
import torch
from torch.nn import Module
import torch.nn.functional as F
        
class ContrastiveLoss(Module):

    """

    Contrastive Loss PyTorch Implementation:

    This class gets a pair of feature maps as (qry_fm, pos_fm) or (qry_fm, neg_fm) and a corresponding target integer value (1 for positive and 0 for negative pairs)
    
    Example: 
    
    loss_fn = ContrastiveLoss(0.5)
    loss_pos = loss_fn(qry_fm, pos_fm, 1.)
    loss_neg = loss_fn(qry_fm, neg_fm, 0.)
    loss = loss_pos + loss_neg

    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        
        self.margin = margin
        self.eps = 1e-9

    def forward(self, fm1, fm2, label, mean=True):
        
        dis = (fm2 - fm1).pow(2).sum(1)  
        losses = 0.5 * (label * dis + (1 + -1 * label) * F.relu(self.margin - (dis + self.eps).sqrt()).pow(2))
        
        return losses.mean() if mean else losses.sum()
# a = torch.rand(3,4,4)
# b = torch.rand(3,4,4)
# loss_fn = ContrastiveLoss(0.5)
# print(loss_fn(a,b,1))
