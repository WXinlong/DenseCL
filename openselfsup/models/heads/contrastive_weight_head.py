import torch
import torch.nn as nn

from ..registry import HEADS


@HEADS.register_module
class ContrastiveWeightHead(nn.Module):
    '''Head for contrastive learning.
    '''

    def __init__(self, temperature=0.1):
        super(ContrastiveWeightHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, pos, neg):
        '''
        Args:
            pos (Tensor): Nx1 positive similarity
            neg (Tensor): Nxk negative similarity
        '''
        N = pos.size(0)
        # relax
        #pos = torch.clamp(pos * 2., max=1.)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        logits[:, 0] *= 2
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        losses = dict()
        losses['loss'] = self.criterion(logits, labels)
        return losses
