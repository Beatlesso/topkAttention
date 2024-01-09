/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include <vector>
#include "cuda/topk_attn_cuda.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>


at::Tensor topk_attn_cuda_forward(
    const at::Tensor &query, 
    const at::Tensor &value,
    const at::Tensor &key,
    const at::Tensor &pos)
{
    AT_ASSERTM(query.is_contiguous(), "query tensor has to be contiguous");
    AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(key.is_contiguous(), "key tensor has to be contiguous");
    AT_ASSERTM(pos.is_contiguous(), "pos tensor has to be contiguous");

    AT_ASSERTM(query.type().is_cuda(), "query must be a CUDA tensor");
    AT_ASSERTM(value.type().is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(key.type().is_cuda(), "key must be a CUDA tensor");
    AT_ASSERTM(pos.type().is_cuda(), "pos must be a CUDA tensor");

}


// 此处多一个 grad_output
std::vector<at::Tensor> topk_attn_cuda_backward(
    const at::Tensor &query, 
    const at::Tensor &value,
    const at::Tensor &key,
    const at::Tensor &pos,
    const at::Tensor &grad_output)
{
    // 检测张量是否内存连续以及是否在cuda上
    AT_ASSERTM(query.is_contiguous(), "query tensor has to be contiguous");
    AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(key.is_contiguous(), "key tensor has to be contiguous");
    AT_ASSERTM(pos.is_contiguous(), "pos tensor has to be contiguous");
    AT_ASSERTM(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    AT_ASSERTM(query.type().is_cuda(), "query must be a CUDA tensor");
    AT_ASSERTM(value.type().is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(key.type().is_cuda(), "key must be a CUDA tensor");
    AT_ASSERTM(pos.type().is_cuda(), "pos must be a CUDA tensor");
    AT_ASSERTM(grad_output.type().is_cuda(), "grad_output must be a CUDA tensor");

}