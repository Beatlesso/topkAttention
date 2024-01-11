from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
# 性能测试使用
from pyinstrument import Profiler
from math import sqrt
import TopkAttention as TKA


class TopkAttnFunction(Function):
    @staticmethod
    def forward(ctx, query, key, value, pos, micro_batch):
        ctx.micro_batch = int(micro_batch)
        pos = pos.type(torch.int32)
        output = TKA.topk_attn_forward(query, key, value, pos, ctx.micro_batch)
        ctx.save_for_backward(query, key, value, pos)
        return output        
    

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        query, key, value, pos = ctx.saved_tensors
        grad_query, grad_key, grad_value, grad_pos = \
            TKA.topk_attn_backward(
                query, key, value, pos, grad_output, ctx.micro_batch)

        return grad_query, grad_key, grad_value, grad_pos


class TopkAttnFunction_Pytorch(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.Attention = ScaledDotProductAttention(scale)

    def forward(self, query, key, value, pos):
        batch, len_q, n_head, ch = query.shape
        topk = pos.shape[-1]

        # 这里需要进行扩张，需要 2*topk 倍的内存
        key = key.unsqueeze(-2).repeat(1, 1, 1, topk, 1)
        value = value.unsqueeze(-2).repeat(1, 1, 1, topk, 1)
        pos = pos.unsqueeze(-1).repeat(1, 1, 1, 1, ch)

        # 这里gather同样需要 2*topk 倍的内存
        # 使用torch.gather
        k = torch.gather(key, 1, pos).reshape(batch, len_q, n_head, topk, ch)
        v = torch.gather(value, 1, pos).reshape(batch, len_q, n_head, topk, ch)

        query = query.unsqueeze(-2)
        output = self.Attention(query, k, v).squeeze(-2)
        return output
    

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        score = torch.matmul(q, k.transpose(-1, -2))
        # print(q.shape)
        # print(k.transpose(-1, -2).shape)
        # print(score)
        # print(score.shape)
        score = score / self.scale 

        attn = self.softmax(score) 
        output = torch.matmul(attn, v) 

        return output
    


class MyAttention3(Function):
    @staticmethod
    def forward(ctx, query, key, value, pos):
        batch, len_q, n_head, ch_qk = query.shape
        ch_v = value.shape[-1]
        topk = pos.shape[-1]

        # 这里需要进行扩张，需要 2*topk 倍的内存
        key = key.unsqueeze(-2).repeat(1, 1, 1, topk, 1)
        value = value.unsqueeze(-2).repeat(1, 1, 1, topk, 1)
        pos_k = pos.unsqueeze(-1).repeat(1, 1, 1, 1, ch_qk)
        pos_v = pos.unsqueeze(-1).repeat(1, 1, 1, 1, ch_v)

        # print(key.shape)
        # print(value.shape)
        # print(pos_k.shape)
        # print(pos_v.shape)

        # 这里gather同样需要 2*topk 倍的内存
        # 使用torch.gather
        k = torch.gather(key, 1, pos_k).reshape(batch, len_q, n_head, topk, ch_qk)
        v = torch.gather(value, 1, pos_v).reshape(batch, len_q, n_head, topk, ch_v)

        query = query.unsqueeze(-2)
        return F.scaled_dot_product_attention(query, k, v).squeeze(-2)
    


# 暴力方案
class Force(Function):
    @staticmethod
    def forward(ctx, query, key, value, pos):
        batch, len_q, n_head, ch_qk = query.shape
        ch_v = value.shape[-1]
        len_kv = value.shape[1]
        device = query.device
        topk = pos.shape[-1]

        # 初始化输出attention结果
        output = torch.zeros(batch, len_q, n_head, ch_v, device=query.device)

        # 直接计算每个query的attention
        for b in range(batch):
            for l in range(len_q):
                for h in range(n_head):
                    # 直接从key和value中提取所需的top-k部分
                    selected_k = key[b, pos[b, l, h], h]
                    selected_v = value[b, pos[b, l, h], h]

                    # 计算scaled dot product attention
                    # 注意：这里假设没有mask，且不需要额外的softmax归一化因子
                    attn_weights = F.softmax(query[b, l, h].matmul(selected_k.transpose(-2, -1)) / sqrt(ch_qk), dim=-1)
                    output[b, l, h] = attn_weights.matmul(selected_v)

        return output