import torch.utils.benchmark as benchmark
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import time
from torch import Tensor
from MyAttention import *
import torch.nn.functional as F
'''
query, key, value  (batch, len=H*W, n_head, d_model / n_head)
pos (batch, H*W, n_head, k)
'''
# 完全注意力batch可以设到160，目前的pytorch实现只能设置到8，这与相差topk倍内存貌似一致
batch = 8
len = 1000
n_head = 8
d_model = 256
topk = 40

# batch = 1
# len = 2
# n_head = 4
# d_model = 32
# topk = 3

dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
torch.cuda.set_device(1)
print(device)

query = torch.randn(batch, len, n_head, d_model // n_head, dtype=dtype, device=device)
key = torch.randn(batch, len, n_head, d_model // n_head, dtype=dtype, device=device)
value = torch.randn(batch, len, n_head, d_model // n_head, dtype=dtype, device=device)
pos = torch.randint(0, len, size=(batch, len, n_head, topk), device=device)

'''
尝试完全注意力区域
'''
# start = time.time()
# out = F.scaled_dot_product_attention(query, key, value)
# end = time.time()
# print('FullAttention Running time  : %s Seconds'%(end-start))
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



# start2 = time.time()
# out2 = MyAttention2.apply(query, key, value, pos)
# # print(out2.shape)
# end2 = time.time()
# print('Running time 2 : %s Seconds'%(end2-start2))


# for i in range(20):
#     print(i)
#     MyAttention2.apply(query, key, value, pos)

# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
#     out2 = MyAttention2.apply(query, key, value, pos)

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
# prof.export_chrome_trace("./PyTorchAttention.json")