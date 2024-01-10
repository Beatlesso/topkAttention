#include <vector>
#include "cuda/topk_attn_cuda.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>


at::Tensor topk_attn_cuda_forward(
    const at::Tensor &query, 
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &pos,
    const int m_batch)
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
    const int len_q = query.size(1);
    const int len_kv = key.size(1);
    const int n_head = query.size(2);
    const int ch_qk = query.size(3);
    const int ch_v = value.size(3);
    const int topk = pos.size(3);
    const int micro_batch = std::min(batch, m_batch);
    AT_ASSERTM(batch % micro_batch == 0, "batch(%d) must divide micro_batch(%d)", batch, micro_batch);
    AT_ASSERTM(ch_qk % ch_v == 0, "the channel of query(%d) must divide the channel of key(%d)", ch_qk, ch_v);
    auto output = at::zeros({batch, len_q, n_head, ch_v}, query.options());
    auto output_view = output.view({batch/micro_batch, micro_batch, len_q, n_head, ch_v});
    auto temp = at::zeros({batch, len_q, n_head, topk}, query.options());
    auto temp_view = temp.view({batch/micro_batch, micro_batch, len_q, n_head, topk});

    /* ************核心部分****************** */

    // 计算每个micro_batch处理的地址偏移
    auto query_strad = micro_batch * len_q * n_head * ch_qk;
    auto key_strad = micro_batch * len_kv * n_head * ch_qk;
    auto value_strad = micro_batch * len_kv * n_head * ch_v;
    auto pos_strad = micro_batch * len_q * n_head * topk;

    for (int n = 0 ; n < batch / micro_batch ; ++ n) 
    {
        auto out_n = output_view.select(0, n);
        auto temp_n = temp_view.select(0, n);
        AT_DISPATCH_FLOATING_TYPES(query.type(), "topk_attention_micro_batch_forward_cuda", ([&] {
            topk_attention_micro_batch_cuda(at::cuda::getCurrentCUDAStream(),     
                query.data<scalar_t>() + n * query_strad,
                key.data<scalar_t>() + n * key_strad,
                value.data<scalar_t>() + n * value_strad,
                pos.data<int>() + n * pos_strad,
                micro_batch, len_q, len_kv, n_head, ch_qk, ch_v, topk,
                temp_n.data<scalar_t>(),
                out_n.data<scalar_t>());
        }));
    }

    output = output.view({batch, len_q, n_head, ch_v});
    /* ************核心部分****************** */

    return output;
}


// 此处y有 grad_output
std::vector<at::Tensor> topk_attn_cuda_backward(
    const at::Tensor &query, 
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &pos,
    const at::Tensor &grad_output,
    const int micro_batch)
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

    // ************核心部分******************

    // ************核心部分******************

    // 反回对应的导数
    return {
        grad_query, grad_value, grad_key, grad_pos
    };
}