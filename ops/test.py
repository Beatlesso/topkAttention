import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from functions.topk_attn_func import TopkAttnFunction, TopkAttnFunction_Pytorch, MyAttention3
import math


batch = 8
len = 1000
n_head = 8
d_model = 256
topk = 20
micro_batch = 1

# batch = 1
# len = 10
# n_head = 8
# d_model = 128
# topk = 5
# micro_batch = 1

dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
torch.cuda.set_device(0)
print(device)

query = torch.randn(batch, len, n_head, d_model // n_head, dtype=dtype, device=device)
key = torch.randn(batch, len, n_head, d_model // n_head, dtype=dtype, device=device)
value = torch.randn(batch, len, n_head, d_model // n_head, dtype=dtype, device=device)
pos = torch.randint(0, len, size=(batch, len, n_head, topk), device=device)

out1 = TopkAttnFunction.apply(query, key, value, pos, micro_batch)
# out1 = MyAttention3.apply(query, key, value, pos)
print(math.sqrt(d_model // n_head))
torch_model = TopkAttnFunction_Pytorch(scale = math.sqrt(d_model // n_head))
out2 = torch_model(query, key, value, pos)
print(out1)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~分隔符~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(out2)
print(out1.dtype)
print(out2.dtype)
print(out1.shape)
print(out2.shape)
assert np.quantile(torch.abs(out1 - out2).cpu(), 0.99) < 5e-5
print("test ok!")

'''

[-10.4350,   4.1009,  -5.6909,  ..., -14.8391, -12.6646,  -3.5315]
[-0.0557,  0.2338, -0.3233,  ..., -0.0485,  0.4762,  0.3452]
'''