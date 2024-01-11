#pragma once

#include "cpu/topk_attn_cpu.h"

#ifdef WITH_CUDA
#include "cuda/topk_attn_cuda.h"
#endif


at::Tensor
topk_attn_forward(
    const at::Tensor &query, 
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &pos,
    const int micro_batch)
{
    if (query.type().is_cuda())
    {
#ifdef WITH_CUDA
        return topk_attn_cuda_forward(
            query, key, value, pos, micro_batch);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor>
topk_attn_backward(
    const at::Tensor &query, 
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &pos,
    const at::Tensor &grad_output,
    const int micro_batch)
{
    if (query.type().is_cuda())
    {
#ifdef WITH_CUDA
        return topk_attn_cuda_backward(
            query, key, value, pos, grad_output, micro_batch);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

