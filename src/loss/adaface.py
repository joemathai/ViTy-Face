# OG version can be found here : https://github.com/mk-minchul/AdaFace/blob/master/head.py
# Combined the partial_fc method with AdaFace
# this code will automatically shard the class centers across GPUs and do per gpu cross-entropy loss 
# by gathering the relavent query samples into the gpu

import torch
import math
import torch.nn as nn
from torch.distributed.nn.functional import _AllGather
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

        # local cross entropy loss
        # TODO: check if distributed cross-entropy makes sense?
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, labels):
        _gather_x = torch.cat(_AllGather.apply(None, x))
        _gather_x_norms = _gather_x.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        _gather_labels = torch.cat(_AllGather.apply(None, labels))
       
        # pick samples that have their weight centers in this shard
        labels = _gather_labels.view(-1, 1)
        index_positive = (self._class_start <= labels) & (labels < self._class_start + self._local_num)
        embeddings = _gather_x[index_positive.squeeze()]
        labels = _gather_labels[index_positive.squeeze()] - self._class_start
        
        norm_embeddings = normalize(embeddings)
        norm_weight_activated = normalize(self._weights)
        cosine = linear(norm_embeddings, norm_weight_activated).clamp(-1, 1)

        safe_norms = torch.clip(_gather_x_norms, min=0.001, max=100).clone().detach()
        with torch.no_grad():
            mean = safe_norms.mean()
            std = safe_norms.std()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms[index_positive.squeeze()] - self.batch_mean) / (self.batch_std + self.eps)
        margin_scaler = margin_scaler * self.h
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        # g_angular
        m_arc = torch.zeros(labels.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, labels.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(labels.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, labels.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        loss = self.ce_loss(scaled_cosine_m, labels)
        return loss

