#pragma once
#include <torch/extension.h>

at::Tensor topk_attn_cuda_forward(
    const at::Tensor &query, 
    const at::Tensor &value,
    const at::Tensor &key,
    const at::Tensor &pos);

std::vector<at::Tensor> topk_attn_cuda_backward(
    const at::Tensor &query, 
    const at::Tensor &value,
    const at::Tensor &key,
    const at::Tensor &pos,
    const at::Tensor &grad_output);
