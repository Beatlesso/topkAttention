#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>


std::vector<at::Tensor>
topk_attn_cpu_forward(
    const at::Tensor &query, 
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &pos,
    const int micro_batch)
{
    AT_ERROR("Not implement on cpu");
}

std::vector<at::Tensor>
topk_attn_cpu_backward(
    const at::Tensor &query, 
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &pos,
    const at::Tensor &grad_output,
    const int micro_batch)
{
    AT_ERROR("Not implement on cpu");
}

