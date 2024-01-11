import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from functions.topk_attn_func import TopkAttnFunction, TopkAttnFunction_Pytorch, MyAttention3
import math
from torch.profiler import profile, record_function, ProfilerActivity

batch = 8
len = 20000
n_head = 8
d_model = 256
topk = 20
micro_batch = 1

# batch = 2
# len = 5
# n_head = 2
# d_model = 8
# topk = 4
# micro_batch = 2

dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
torch.cuda.set_device(0)
print(device)

query = torch.randn(batch, len, n_head, d_model // n_head, dtype=dtype, device=device)
key = torch.randn(batch, len, n_head, d_model // n_head, dtype=dtype, device=device)
value = torch.randn(batch, len, n_head, d_model // n_head, dtype=dtype, device=device)
pos = torch.randint(0, len, size=(batch, len, n_head, topk), device=device)


'''
下面是正确性测试
'''
# print(query)
# print(key)
# print(value)
# print(pos)
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# out1 = TopkAttnFunction.apply(query, key, value, pos, micro_batch)

# # torch_model = TopkAttnFunction_Pytorch(scale = math.sqrt(d_model // n_head))
# # out2 = torch_model(query, key, value, pos)
# out2 = MyAttention3.apply(query, key, value, pos)
# # print(out1)
# # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# # print("~~~~~~~~~~~~~分隔符~~~~~~~~~~~~~")
# # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# # print(out2)
# # # print(out1.dtype)
# # # print(out2.dtype)
# # # print(out1.shape)
# # # print(out2.shape)
# assert np.quantile(torch.abs(out1 - out2).cpu(), 0.99) < 5e-5
# print("test ok!")




'''
下面是性能测试
'''
# CUDA
for i in range(20):
    TopkAttnFunction.apply(query, key, value, pos, micro_batch)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
    out1 = TopkAttnFunction.apply(query, key, value, pos, micro_batch)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
prof.export_chrome_trace("./TopkAttnFunction.json")

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~分隔符~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

#  Pytorch
for i in range(20):
    MyAttention3.apply(query, key, value, pos)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
    out1 = MyAttention3.apply(query, key, value, pos)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
prof.export_chrome_trace("./PytorchAttnFunction.json")