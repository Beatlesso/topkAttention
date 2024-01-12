import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from functions.topk_attn_func import *
from torch.profiler import profile, record_function, ProfilerActivity


# batch = 8
# len = 2000
# n_head = 8
# d_model = 256
# topk = 20
# micro_batch = 8

# # batch = 1
# # len = 2
# # n_head = 4
# # d_model = 32
# # topk = 3

# dtype = torch.float32
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
# torch.cuda.set_device(0)
# print(device)

# query = torch.randn(batch, len, n_head, d_model // n_head, dtype=dtype, device=device)
# key = torch.randn(batch, len, n_head, d_model // n_head, dtype=dtype, device=device)
# value = torch.randn(batch, len, n_head, d_model // n_head, dtype=dtype, device=device)
# pos = torch.randint(0, len, size=(batch, len, n_head, topk), device=device)


'''
适配不同的len_q和len_kv, 以及不同的ch_qk和ch_v测试
'''
# batch = 8
# len_q = 1000
# len_kv = 1000
# n_head = 8
# d_model_qk = 256
# d_model_v = 256
# topk = 20
# micro_batch = 8

batch = 2
len_q = 10
len_kv = 10
n_head = 2
d_model_qk = 8
d_model_v = 8
topk = 5
micro_batch = 2

dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
torch.cuda.set_device(0)
print(device)

query = torch.randn(batch, len_q, n_head, d_model_qk // n_head, dtype=dtype, device=device)
key = torch.randn(batch, len_kv, n_head, d_model_qk // n_head, dtype=dtype, device=device)
value = torch.randn(batch, len_kv, n_head, d_model_v // n_head, dtype=dtype, device=device)
pos = torch.randint(0, len_kv, size=(batch, len_q, n_head, topk), device=device)
query.requires_grad_(True)
key.requires_grad_(True)
value.requires_grad_(True)



'''
下面是正确性测试
'''
# # print(query)
# # print(key)
# # print(value)
# # print(pos)
# # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
out1 = TopkAttnFunction.apply(query, key, value, pos, micro_batch)
out1.sum().backward()
grad_query_1 = query.grad.clone()
grad_key_1 = key.grad.clone()
grad_value_1 = value.grad.clone()
query.grad.zero_()
key.grad.zero_()
value.grad.zero_()

model_2 = MyAttention3()
out2 = model_2(query, key, value, pos)
out2.sum().backward()
grad_query_2 = query.grad
grad_key_2 = key.grad
grad_value_2 = value.grad


print(grad_key_1)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~分隔符~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(grad_key_2)

# # torch_model = TopkAttnFunction_Pytorch(scale = sqrt(d_model_qk // n_head))
# # out2, attn2 = torch_model(query, key, value, pos)
# # out2 = MyAttention3.apply(query, key, value, pos)
# out2, attn2 = Force.apply(query, key, value, pos)
# # print(out1.shape)
# # print(out1)
# # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# # print("~~~~~~~~~~~~~分隔符~~~~~~~~~~~~~")
# # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# # print(out2.shape)
# # print(out2)
# # # print(out1.dtype)
# # # print(out2.dtype)
# # # print(out1.shape)
# # # print(out2.shape)
# assert np.quantile(torch.abs(out1 - out2).cpu().detach().numpy(), 0.99) < 5e-5
# assert np.quantile(torch.abs(attn1 - attn2).cpu().detach().numpy(), 0.99) < 5e-5
assert np.quantile(torch.abs(grad_value_1 - grad_value_2).cpu().detach().numpy(), 0.99) < 5e-5
assert np.quantile(torch.abs(grad_key_1 - grad_key_2).cpu().detach().numpy(), 0.99) < 5e-5
assert np.quantile(torch.abs(grad_value_1 - grad_value_2).cpu().detach().numpy(), 0.99) < 5e-5
print("test ok!")




'''
下面是性能测试
'''
# # CUDA
# for i in range(20):
#     TopkAttnFunction.apply(query, key, value, pos, micro_batch)
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
#     out, attn = TopkAttnFunction.apply(query, key, value, pos, micro_batch)

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
# prof.export_chrome_trace("./TopkAttnFunction.json")

# torch.cuda.synchronize()
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