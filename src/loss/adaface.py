# OG version can be found here : https://github.com/mk-minchul/AdaFace/blob/master/head.py
# Combined the partial_fc method with AdaFace
# Assumption: All the GPUs sees the exact same batch. (not very compute efficient)
# Each rank computes the backbone and a part of the final classification layer. 
# A reduce operation is used obtain the sum of exp of logits needed for the denominator of the cross-entropy loss
# Each rank will compute the cross entropy loss for the samples whose weight centers are in the local rank (numerator of CELoss)

import torch
import math
import torch.nn as nn
import torch.distributed as dist
from torch.nn.functional import linear, normalize


class PartialFC_AdaFace(nn.Module):
    def __init__(
        self,
        embedding_size=512,
        classnum=70722,
        rank=0,
        world_size=1,
        m=0.4,
        h=0.333,
        s=64.,
        t_alpha=1.0,
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.embed_size = embedding_size

        # make num classes divisible by world_size
        self._num_classes = int(math.ceil(classnum / self.world_size) * self.world_size)
        self._local_num = self._num_classes // self.world_size
        self._class_start = self._num_classes // self.world_size * self.rank

        # split the weight matrix 
        self._weights = nn.Parameter(torch.normal(0, 0.01, (self._local_num, self.embed_size)))
        
        # params for AdaFace
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1) * 20)
        self.register_buffer('batch_std', torch.ones(1) * 100)

    def forward(self, x, labels):
        x_norms = x.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        labels = labels.view(-1, 1)
        
        # index of samples whose weight centers are in this rank
        index_positive = (self._class_start <= labels) & (labels < self._class_start + self._local_num)

        norm_embeddings = normalize(x)
        norm_weight_activated = normalize(self._weights)
        cosine = linear(norm_embeddings, norm_weight_activated).clamp(-1, 1)

        safe_norms = torch.clip(x_norms, min=0.001, max=100)
        with torch.no_grad():
            mean = safe_norms.mean()
            std = safe_norms.std()
            dist.all_reduce(mean, dist.ReduceOp.SUM)
            dist.all_reduce(std, dist.ReduceOp.SUM)
            # avg of avg (same subsets)
            mean = mean / dist.get_world_size()
            std = std / dist.get_world_size()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler = margin_scaler * self.h
        margin_scaler = torch.clip(margin_scaler, -1, 1)
       
        # g_angular, g_additive and scaling
        margin_final_logit = torch.cos(torch.arccos(cosine) + (self.m * margin_scaler * -1))
        scaled_cosine_m = self.s * (margin_final_logit - (self.m + (self.m * margin_scaler)))

        # distributed cross-entropy loss
        logits_exp = torch.exp(scaled_cosine_m)
        # calculate the common denominator for the distributed cross-entropy loss
        sum_logits_exp = torch.sum(logits_exp, dim=1, keepdim=True)
        dist.all_reduce(sum_logits_exp, dist.ReduceOp.SUM)
        
        # compute the cross-entropy loss for this shard
        loss = logits_exp[index_positive.squeeze()] / sum_logits_exp[index_positive.squeeze()]
        # FIXME: this is hack if the bacth doesn't have samples whose weight are present here,
        # pass a very small random noise as gradient in this case
        if loss.nelement() == 0: 
            return scaled_cosine_m.mean() * 1e-30

        ce_loss = loss.clamp_min(1e-30).log_().mean() * -1
        return ce_loss


if __name__ == "__main__":
    # unit-test 
    # torchrun --nnodes=1 --nproc-per-node=2 loss/adaface.py
    dist.init_process_group(backend="nccl")
    embedding = torch.randn(10, 1536).to(dist.get_rank())
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).to(dist.get_rank())
    model = PartialFC_AdaFace(embedding_size=1536, rank=dist.get_rank(), world_size=dist.get_world_size()).to(dist.get_rank())
    opt = torch.optim.Adam(model.parameters(), lr=1e1)
    for i in range(10):
        opt.zero_grad()
        y = model(embedding, labels)
        if y: 
            y.backward()
            opt.step()
        print(y)

