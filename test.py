import torch.utils.benchmark as benchmark
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import time
from torch import Tensor
from MyAttention import *
import torch.nn.functional as F
from Attention import *
'''
query, key, value  (batch, len=H*W, n_head, d_model / n_head)
pos (batch, H*W, n_head, k)
'''
# 完全注意力batch可以设到160+，目前的pytorch实现只能设置到8，这与相差topk倍内存貌似一致
# 第二个实现不需要额外显存，batch也可以设置到160+，但是速度太慢了
'''
上述设置其余参数如下：
len = 20000
n_head = 8
d_model = 256
topk = 20
'''

batch = 8
len = 1000
n_head = 8
d_model = 256
topk = 20

# batch = 1
# len = 2
# n_head = 4
# d_model = 32
# topk = 3

dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
torch.cuda.set_device(0)
print(device)

query = torch.randn(batch, len, n_head, d_model // n_head, dtype=dtype, device=device)
key = torch.randn(batch, len, n_head, d_model // n_head, dtype=dtype, device=device)
value = torch.randn(batch, len, n_head, d_model // n_head, dtype=dtype, device=device)

# q = [[[[5]], [[3]], [[6]], [[4]]]]
# k = [[[[3]], [[7]], [[5]], [[1]]]]
# v = [[[[6]], [[5]], [[2]], [[8]]]]
# query = torch.tensor(q, dtype=dtype)
# key = torch.tensor(k, dtype=dtype)
# value = torch.tensor(v, dtype=dtype)
# print(query)

pos = torch.randint(0, len, size=(batch, len, n_head, topk), device=device)

'''
尝试完全注意力区域
对于完全注意力而言，qkv的形状必须为 (batch, n_head, H*W, d_model / n_head) 才是正确的
'''
# start = time.time()
# query = query.transpose(1, 2)
# key = key.transpose(1, 2)
# value = value.transpose(1, 2)


# # print(value.shape)
# out = F.scaled_dot_product_attention(query, key, value)
# # print(out.shape)
# # print(out)
# end = time.time()
# print('FullAttention Running time  : %s Seconds'%(end-start))
# scale = np.sqrt(d_model // n_head)
# Attention = ScaledDotProductAttention(scale)
# attn, std_out = Attention(query, key, value, None)
# assert np.quantile(torch.abs(out - std_out).cpu(), 0.99) < 5e-5
# print('check OK!')


# # output = torch.zeros(1, 1, 4, 1)
# # for i in range(4):
# #     attn_weights = F.softmax(query[0, 0, i].matmul(key[0][0].transpose(-2, -1)) / sqrt(1), dim=-1)
# #     print(attn_weights)
# #     output[0, 0, i] = attn_weights.matmul(value[0][0])
# # print(output)
'''
尝试完全注意力区域
'''

start1 = time.time()
out1 = MyAttention1.apply(query, key, value, pos)
# print(out1.shape)
torch.cuda.synchronize()
end1 = time.time()
print('test 1 run ok')

# time.sleep(10)

start2 = time.time()
out2 = MyAttention2.apply(query, key, value, pos)
# print(out2.shape)
torch.cuda.synchronize()
end2 = time.time()
print('test 2 run ok')


start3 = time.time()
out3 = MyAttention3.apply(query, key, value, pos)
# print(out3.shape)
torch.cuda.synchronize()
end3 = time.time()
print('test 3 run ok')

assert np.quantile(torch.abs(out1 - out2).cpu(), 0.99) < 5e-5
assert np.quantile(torch.abs(out1 - out3).cpu(), 0.99) < 5e-5
assert np.quantile(torch.abs(out2 - out3).cpu(), 0.99) < 5e-5
print("check ok!")


print('Running time 1 : %s Seconds'%(end1-start1))
print('Running time 2 : %s Seconds'%(end2-start2))
print('Running time 3 : %s Seconds'%(end3-start3))

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

# start2 = time.time()
# out2 = MyAttention2.apply(query, key, value, pos)
# # print(out2.shape)
# end2 = time.time()
# print('Running time 2 : %s Seconds'%(end2-start2))

# start3 = time.time()
# out3 = MyAttention3.apply(query, key, value, pos)
# end3 = time.time()
# print('Running time 3 : %s Seconds'%(end3-start3))



# for i in range(20):
#     print(i)
#     MyAttention2.apply(query, key, value, pos)

# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
#     out2 = MyAttention2.apply(query, key, value, pos)

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
# prof.export_chrome_trace("./PyTorchAttention.json")