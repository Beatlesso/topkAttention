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

    // query, key, value  (batch, len=H*W, n_head, d_model / n_head)
    // pos (batch, len=H*W, n_head, k)
    // 拿出对应的参数
    const int batch = query.size(0);
    const int len = query.size(1);
    const int n_head = query.size(2);
    const int channels = query.size(3);
    const int topk = pos.size(3);

    auto output = at::zeros({batch, len, n_head, channels}, query.options());




    return output;
}


// 此处y有 grad_output
std::vector<at::Tensor> topk_attn_cuda_backward(
    const at::Tensor &query, 
    const at::Tensor &value,
    const at::Tensor &key,
    const at::Tensor &pos,
    const at::Tensor &grad_output)
{
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

    // query, key, value  (batch, len=H*W, n_head, d_model / n_head)
    // pos (batch, len=H*W, n_head, k)
    // 拿出对应的参数
    const int batch = query.size(0);
    const int len = query.size(1);
    const int n_head = query.size(2);
    const int channels = query.size(3);
    const int topk = pos.size(3);

    auto grad_query = at::zeros_like(query);
    auto grad_value = at::zeros_like(value);
    auto grad_key = at::zeros_like(key);
    auto grad_pos = at::zeros_like(pos);

    // 反回对应的导数
    return {
        grad_query, grad_value, grad_key, grad_pos
    };
}