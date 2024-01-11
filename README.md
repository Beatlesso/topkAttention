# topkAttention
测试数据规模如下：
```
batch = 160
len = 20000
n_head = 8
d_model = 256
topk = 20
micro_batch = 8
```


目前pytorch版本的实现在MyAttention.py中，总共三个版本，目前测试三个结果是一致的
第一个版本和暴力无差别，最能保证正确性，需要额外topk倍的显存，速度很慢，在batch为8的情况下需要30s
第二个版本避免了显示存储k和v，避免了额外的显存，batch可以设置到160，但是速度比第一个版本还要慢，在batch为8的情况下需要60s
第三个版本速度最快，但是仍然无法避免topk倍的额外显存占用，batch最大只能设置为8，但是速度很快，只需要0.11s（此处结果由于异步的原因导致错误）

测试使用的代码在test.py中


2024/1/11：目前已经实现了topkAttention的forward，其速度相比pytorch的实现速度要快很多（CUDA版本16.241ms，Pytorch实现1.057s），且不需要额外topk倍的内存。
目前准备写反向传播的时候，发现了问题，当前的forward没有存储用于反向传播所需要的attn矩阵，需要改成对应的形式，可能速度会变慢，但应该仍然比Pytorch速度快。
将线程块大小优化成了 
```block_size = (n_head * ch_v <= 1024 ? n_head * ch_v : ch_v)```
速度由原来的16.241ms进一步降低到了12.994ms

详细对比
FullAttention:
Self CPU time total: 5.484s, 但是其中cudaDeviceSynchronize 占了 99.94% 共计 5.480s
Self CUDA time total: 261.267ms

TopkAttnFunction_Pytorch(需要额外topk倍显存):
Self CPU time total: 1.057s, 但是其中cudaDeviceSynchronize 占了 94.5% 共计 998.616ms
Self CUDA time total: 58.698ms

CUDA_TopkAttnFunction:
Self CPU time total: 12.914ms, 但是其中cudaDeviceSynchronize 占了 82.37% 共计 10.637ms
Self CUDA time total: 10.975ms

修复了无法正确支持ch_qk和ch_v不同的情况，现已能够正常支持ch_qk和ch_v不同的情况，同时len_q和len_kv不同的情况也是支持的。
