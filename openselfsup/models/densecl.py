import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS


@MODELS.register_module
class DenseCL(nn.Module):
    '''DenseCL.
    Part of the code is borrowed from:
        "https://github.com/facebookresearch/moco/blob/master/moco/builder.py".
    '''

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 loss_lambda=0.5,
                 **kwargs):
        super(DenseCL, self).__init__()
        self.encoder_q = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.encoder_k = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.backbone = self.encoder_q[0]
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

        self.queue_len = queue_len
        self.momentum = momentum
        self.loss_lambda = loss_lambda

        # create the queue
        self.register_buffer("queue", torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # create the second queue for dense output
        self.register_buffer("queue2", torch.randn(feat_dim, queue_len))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)
        self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.encoder_q[0].init_weights(pretrained=pretrained)
        self.encoder_q[1].init_weights(init_linear='kaiming')
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue2_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue2_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward_train(self, img, **kwargs):
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        im_q = img[:, 0, ...].contiguous()
        im_k = img[:, 1, ...].contiguous()
        # compute query features
        q_b = self.encoder_q[0](im_q) # backbone features
        q, q_grid, q2 = self.encoder_q[1](q_b)  # queries: NxC; NxCxS^2
        q_b = q_b[0]
        q_b = q_b.view(q_b.size(0), q_b.size(1), -1)

        q = nn.functional.normalize(q, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)
        q_grid = nn.functional.normalize(q_grid, dim=1)
        q_b = nn.functional.normalize(q_b, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k_b = self.encoder_k[0](im_k)
            k, k_grid, k2 = self.encoder_k[1](k_b)  # keys: NxC; NxCxS^2
            k_b = k_b[0]
            k_b = k_b.view(k_b.size(0), k_b.size(1), -1)

            k = nn.functional.normalize(k, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)
            k_grid = nn.functional.normalize(k_grid, dim=1)
            k_b = nn.functional.normalize(k_b, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k2 = self._batch_unshuffle_ddp(k2, idx_unshuffle)
            k_grid = self._batch_unshuffle_ddp(k_grid, idx_unshuffle)
            k_b = self._batch_unshuffle_ddp(k_b, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # feat point set sim
        backbone_sim_matrix = torch.matmul(q_b.permute(0, 2, 1), k_b)
        densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1] # NxS^2

        indexed_k_grid = torch.gather(k_grid, 2, densecl_sim_ind.unsqueeze(1).expand(-1, k_grid.size(1), -1)) # NxCxS^2
        densecl_sim_q = (q_grid * indexed_k_grid).sum(1) # NxS^2

        l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1) # NS^2X1

        q_grid = q_grid.permute(0, 2, 1)
        q_grid = q_grid.reshape(-1, q_grid.size(2))
        l_neg_dense = torch.einsum('nc,ck->nk', [q_grid,
                                            self.queue2.clone().detach()])

        loss_single = self.head(l_pos, l_neg)['loss_contra']
        loss_dense = self.head(l_pos_dense, l_neg_dense)['loss_contra']

        losses = dict()
        losses['loss_contra_single'] = loss_single * (1 - self.loss_lambda)
        losses['loss_contra_dense'] = loss_dense * self.loss_lambda

        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue2(k2)

        return losses

    def forward_test(self, img, **kwargs):
        im_q = img.contiguous()
        # compute query features
        #_, q_grid, _ = self.encoder_q(im_q)
        q_grid = self.backbone(im_q)[0]
        q_grid = q_grid.view(q_grid.size(0), q_grid.size(1), -1)
        q_grid = nn.functional.normalize(q_grid, dim=1)
        return None, q_grid, None

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
