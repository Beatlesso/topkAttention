#pragma once
#include <torch/extension.h>

at::Tensor topk_attn_cuda_forward(
    const at::Tensor &query, 
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &pos,
    const int micro_batch);

std::vector<at::Tensor> topk_attn_cuda_backward(
    const at::Tensor &query, 
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &pos,
    const at::Tensor &grad_output,
    const int micro_batch);
