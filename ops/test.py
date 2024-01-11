import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from functions.topk_attn_func import *
from torch.profiler import profile, record_function, ProfilerActivity

batch = 8
len_q = 100
len_kv = 200
n_head = 8
d_model_qk = 256
d_model_v = 128
topk = 20
micro_batch = 8

# batch = 2
# len = 10
# n_head = 2
# d_model_qk = 8
# d_model_v = 4
# topk = 5
# micro_batch = 2

dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
torch.cuda.set_device(0)
print(device)

query = torch.randn(batch, len_q, n_head, d_model_qk // n_head, dtype=dtype, device=device)
key = torch.randn(batch, len_kv, n_head, d_model_qk // n_head, dtype=dtype, device=device)
value = torch.randn(batch, len_kv, n_head, d_model_v // n_head, dtype=dtype, device=device)
pos = torch.randint(0, len_kv, size=(batch, len_q, n_head, topk), device=device)


'''
下面是正确性测试
'''
print(query)
print(key)
print(value)
print(pos)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

out1 = TopkAttnFunction.apply(query, key, value, pos, micro_batch)
# out1 = Force.apply(query, key, value, pos)

# torch_model = TopkAttnFunction_Pytorch(scale = math.sqrt(d_model // n_head))
# out2 = torch_model(query, key, value, pos)
# out2 = MyAttention3.apply(query, key, value, pos)
out2 = Force.apply(query, key, value, pos)
print(out1.shape)
print(out1)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~分隔符~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(out2.shape)
print(out2)
# # print(out1.dtype)
# # print(out2.dtype)
# # print(out1.shape)
# # print(out2.shape)
assert np.quantile(torch.abs(out1 - out2).cpu(), 0.99) < 5e-5
print("test ok!")




'''
下面是性能测试
'''
# # CUDA
# for i in range(20):
#     TopkAttnFunction.apply(query, key, value, pos, micro_batch)
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
#     out = TopkAttnFunction.apply(query, key, value, pos, micro_batch)

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
# prof.export_chrome_trace("./TopkAttnFunction.json")

# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# print("~~~~~~~~~~~~~分隔符~~~~~~~~~~~~~")
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# #  Pytorch
# with torch.no_grad():
#     for i in range(20):
#         MyAttention3.apply(query, key, value, pos)
#     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
#         out = MyAttention3.apply(query, key, value, pos)

#     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
#     prof.export_chrome_trace("./PytorchAttnFunction.json")



# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# print("~~~~~~~~~~~~~分隔符~~~~~~~~~~~~~")
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# #  FullAttention
# with torch.no_grad():
#     query = query.transpose(1, 2)
#     key = key.transpose(1, 2)
#     value = value.transpose(1, 2)
#     for i in range(20):
#         out = F.scaled_dot_product_attention(query, key, value)
#     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
#         out = F.scaled_dot_product_attention(query, key, value)

#     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
#     prof.export_chrome_trace("./FullAttention.json")