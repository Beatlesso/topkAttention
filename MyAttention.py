import torch
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn as nn
from math import sqrt
import einops
'''
既然是假设top-k已经计算好，那么这边就直接拿query，key，value作为输入
先暂定三者的维度一致，都为 (batch, H*W, n_head, d_model / n_head)
pos (batch, H*W, n_head, k) 表示对应每个query要去计算的top-k的key和value坐标
不过这里需要考量pos中究竟是存位置还是直接存对应的值，先假设存位置

F.scaled_dot_product_attention(query, key, value)
假设 query(N, ..., L, E)
     key(N, ..., S, E)
     value(N, ..., S, Ev)
那么输出为 output(N, ..., L, Ev)

output = MyAttention.apply(query, key, value, pos)
'''

# 方案1  暴力方案，保证正确性
class MyAttention1(Function):
    @staticmethod
    def forward(ctx, query, key, value, pos):
        batch, len, n_head, ch = query.shape
        device = query.device
        topk = pos.shape[-1]
        # 这里相当于把每个query要计算的topk个key和value存下来，需要额外 topk倍的内存
        k = torch.zeros(batch, len, n_head, topk, ch, device=device)
        v = torch.zeros(batch, len, n_head, topk, ch, device=device)
        query = query.unsqueeze(-2)

        # 使用对应的key和value引填充k和v
        for b in range(batch):
            for l in range(len):
                for h in range(n_head):
                    k[b, l, h] = key[b, pos[b, l, h], h]
                    v[b, l, h] = value[b, pos[b, l, h], h]

        print(k.shape)
        print(v.shape)

        return F.scaled_dot_product_attention(query, k, v).squeeze(-2)
    

# 方案2  避免显示存储k和v导致的 额外topk倍的内存
class MyAttention2(Function):
    @staticmethod
    def forward(ctx, query, key, value, pos):
        batch, len, n_head, ch = query.shape
        topk = pos.shape[-2]

        # 初始化输出attention结果
        output = torch.zeros(batch, len, n_head, ch, device=query.device)


        # 直接计算每个query的attention
        for b in range(batch):
            for l in range(len):
                for h in range(n_head):
                    # 直接从key和value中提取所需的top-k部分
                    selected_k = key[b, pos[b, l, h], h]
                    selected_v = value[b, pos[b, l, h], h]

                    # 计算scaled dot product attention
                    # 注意：这里假设没有mask，且不需要额外的softmax归一化因子
                    attn_weights = F.softmax(query[b, l, h].matmul(selected_k.transpose(-2, -1)) / sqrt(ch), dim=-1)
                    output[b, l, h] = attn_weights.matmul(selected_v)

        return output


# 方案3 效率方案，保证速度
class MyAttention3(Function):
    @staticmethod
    def forward(ctx, query, key, value, pos):
        batch, len_q, n_head, ch = query.shape
        topk = pos.shape[-1]

        # 这里需要进行扩张，需要 2*topk 倍的内存
        key = key.unsqueeze(-2).repeat(1, 1, 1, topk, 1)
        value = value.unsqueeze(-2).repeat(1, 1, 1, topk, 1)
        pos = pos.unsqueeze(-1).repeat(1, 1, 1, 1, ch)

        # print(key.shape)
        # print(value.shape)
        # print(pos.shape)

        # 这里gather同样需要 2*topk 倍的内存
        # 使用torch.gather
        k = torch.gather(key, 1, pos).reshape(batch, len_q, n_head, topk, ch)
        v = torch.gather(value, 1, pos).reshape(batch, len_q, n_head, topk, ch)

        print(k.shape)
        print(v.shape)

        query = query.unsqueeze(-2)
        return F.scaled_dot_product_attention(query, k, v).squeeze(-2)

