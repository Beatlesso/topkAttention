#pragma once

#include "cpu/topk_attn_cpu.h"

#ifdef WITH_CUDA
#include "cuda/topk_attn_cuda.h"
#endif


at::Tensor
topk_attn_forward(
    const at::Tensor &query, 
    const at::Tensor &value,
    const at::Tensor &key,
    const at::Tensor &pos)
{
    if (value.type().is_cuda())
    {
#ifdef WITH_CUDA
        return topk_attn_cuda_forward(
            query, value, key, pos);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor>
topk_attn_backward(
    const at::Tensor &query, 
    const at::Tensor &value,
    const at::Tensor &key,
    const at::Tensor &pos,
    const at::Tensor &grad_output)
{
    if (value.type().is_cuda())
    {
#ifdef WITH_CUDA
        return topk_attn_cuda_backward(
            query, value, key, pos, grad_output);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

