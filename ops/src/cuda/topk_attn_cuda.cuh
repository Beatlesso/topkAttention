#include <cstdio>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THCAtomics.cuh>

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int MAXN = 1024;
const int MAXN_HEAD = 64;
inline int GET_BLOCKS(const int N, const int num_threads)
{
  return (N + num_threads - 1) / num_threads;
}



template <typename scalar_t>
__global__ void topk_attention_micro_batch_forward_gpu_kernel(  const int n, // n = micro_batch * len_q * n_head * ch_v = element(output)
                                                        const scalar_t* query,  //query (micro_batch, len_q,  n_head, ch_qk)
                                                        const scalar_t* key,    //key   (micro_batch, len_kv, n_head, ch_qk )
                                                        const scalar_t* value,  //value (micro_batch, len_kv, n_head, ch_v )
                                                        const int* pos, //pos (micro_batch, len_q, n_head, topk)
                                                        const int micro_batch, 
                                                        const int len_q, 
                                                        const int len_kv, 
                                                        const int n_head, 
                                                        const int ch_qk, 
                                                        const int ch_v, 
                                                        const int topk,
                                                        scalar_t* output, // output (micro_batch, len_q, n_head, ch_v)
                                                        scalar_t* attn)
{
    /*
        #define CUDA_KERNEL_LOOP(i, n)                          \
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
            i < (n);                                          \
            i += blockDim.x * gridDim.x)
        一个index对应输出的一个数
        只要num_kernels和num_actual_kernels相等，这个循环就只会执行一次
        output(micro_batch, len_q, n_head, ch_v)

        此实现假定 len_q = len_kv, ch_qk = ch_v
    */

    const double fact = sqrt(ch_qk);
    CUDA_KERNEL_LOOP(index, n) 
    {
        int idx = index;
        const int ch_idx = idx % ch_v;
        idx /= ch_v;
        const int head_idx = idx % n_head;
        idx /= n_head;
        const int q_idx = idx % len_q;
        idx /= len_q;
        const int batch_idx = idx;  

        const int thread_id = threadIdx.x % ch_v;
        const int thread_head_id = threadIdx.x / ch_v;

        const int query_head_strad = ch_qk;
        const int query_len_strad = n_head * query_head_strad;
        const int query_batch_strad = len_q * query_len_strad;

        const int key_head_strad = ch_qk;
        const int key_len_strad = n_head * key_head_strad;
        const int key_batch_strad = len_kv * key_len_strad;

        const int value_head_strad = ch_v;
        const int value_len_strad = n_head * value_head_strad;
        const int value_batch_strad = len_kv * value_len_strad;

        const int pos_head_strad = topk;
        const int pos_len_strad = n_head * pos_head_strad;
        const int pos_batch_strad = len_q * pos_len_strad;

        const int output_head_strad = ch_v;
        const int output_len_strad = n_head * output_head_strad;
        const int output_batch_strad = len_q * output_len_strad;

        const int attn_head_strad = topk;
        const int attn_len_strad = n_head * attn_head_strad;
        const int attn_batch_strad = len_q * attn_len_strad;

        __shared__ scalar_t temp_buffer[MAXN];
        const int thread_temp_buffer_offset = thread_head_id * topk;
        for(int i = thread_id ; i < topk ; i += ch_v) {
            temp_buffer[i + thread_temp_buffer_offset] = 0;
        }
        __syncthreads();

        const int* pos_ptr = pos + batch_idx * pos_batch_strad + q_idx * pos_len_strad + head_idx * pos_head_strad; 
        const scalar_t* query_ptr = query + batch_idx * query_batch_strad + q_idx * query_len_strad + head_idx * query_head_strad;
        for (int t = 0 ; t < topk ; ++ t) 
        {
            const int position = pos_ptr[t];
            const scalar_t* key_ptr = key + batch_idx * key_batch_strad + position * key_len_strad + head_idx * key_head_strad;
            scalar_t thread_sum = 0;
            for(int j = thread_id ; j < ch_qk ; j += ch_v)
            {
              thread_sum += query_ptr[j] * key_ptr[j];
              if(q_idx == 0 && head_idx == 0) 
              {
                    // printf("batch_idx is %d, q_idx is %d, head_idx is %d, ch_idx is %d\n  \
                    //     t is %d, pos is %d, query is %lf, key is %lf\n",  \
                    //     batch_idx, q_idx, head_idx, ch_idx,     \
                    //     t, position, query_ptr[j], key_ptr[j]);
              }
            } 
            atomicAdd(&temp_buffer[thread_temp_buffer_offset + t], thread_sum);
        }
        __syncthreads();

        // if(head_idx == 0 && q_idx == 0 && ch_idx == 0) {
        //     for (int t = 0 ; t < topk ; ++ t) 
        //     {
        //         printf("temp_buffer[%d] is %lf ", t, temp_buffer[t]);
        //     }
        //     printf("\n");
        // }        

        __shared__ scalar_t sh_maxv[MAXN_HEAD], sh_sum[MAXN_HEAD];
        if (thread_id == 0) {
            sh_maxv[thread_head_id] = -FLT_MAX;
            sh_sum[thread_head_id] = 0;
        }

        for(int i = thread_id ; i < topk ; i += ch_v) {
            temp_buffer[i + thread_temp_buffer_offset] /= fact;
            sh_maxv[thread_head_id] = max(sh_maxv[thread_head_id], temp_buffer[i + thread_temp_buffer_offset]);
        }
        __syncthreads();


        for(int i = thread_id ; i < topk ; i += ch_v) {
            temp_buffer[i + thread_temp_buffer_offset] = exp(temp_buffer[i + thread_temp_buffer_offset] - sh_maxv[thread_head_id]);
            atomicAdd(&sh_sum[thread_head_id], temp_buffer[i + thread_temp_buffer_offset]);
        }
        __syncthreads();


        for(int i = thread_id ; i < topk ; i += ch_v) {
            temp_buffer[i + thread_temp_buffer_offset] /= sh_sum[thread_head_id];
        }
        __syncthreads();

        scalar_t* attn_ptr = attn + batch_idx * attn_batch_strad + q_idx * attn_len_strad + head_idx * attn_head_strad;
        for(int i = thread_id ; i < topk ; i += ch_v) {
            attn_ptr[i] = temp_buffer[i + thread_temp_buffer_offset];
        }

        scalar_t* output_ptr = output + batch_idx * output_batch_strad + q_idx * output_len_strad + head_idx * output_head_strad;
        for (int t = 0 ; t < topk ; ++ t) 
        {
            const int position = pos_ptr[t];
            const scalar_t* value_ptr = value + batch_idx * value_batch_strad + position * value_len_strad + head_idx * value_head_strad;  
            for (int j = thread_id ; j < ch_v ; j += ch_v)
            {
                atomicAdd(&output_ptr[j], value_ptr[j] * temp_buffer[t + thread_temp_buffer_offset]);
            }
        }  
    }
}




template <typename scalar_t>
__global__ void topk_attention_micro_batch_backward_gpu_kernel(  const int n, // n = micro_batch * len_q * n_head * ch_v = element(output)
                                                        const scalar_t* query,  //query (micro_batch, len_q,  n_head, ch_qk)
                                                        const scalar_t* key,    //key   (micro_batch, len_kv, n_head, ch_qk )
                                                        const scalar_t* value,  //value (micro_batch, len_kv, n_head, ch_v )
                                                        const int* pos, //pos (micro_batch, len_q, n_head, topk)
                                                        const scalar_t* attn,
                                                        const scalar_t* grad_output,
                                                        const int micro_batch, 
                                                        const int len_q, 
                                                        const int len_kv, 
                                                        const int n_head, 
                                                        const int ch_qk, 
                                                        const int ch_v, 
                                                        const int topk,
                                                        scalar_t* grad_query, 
                                                        scalar_t* grad_key,
                                                        scalar_t* grad_value)
{
    const double fact = sqrt(ch_qk);
    CUDA_KERNEL_LOOP(index, n) 
    {
        int idx = index;
        const int ch_idx = idx % ch_v;
        idx /= ch_v;
        const int head_idx = idx % n_head;
        idx /= n_head;
        const int q_idx = idx % len_q;
        idx /= len_q;
        const int batch_idx = idx;  

        const int thread_id = threadIdx.x % ch_v;
        const int thread_head_id = threadIdx.x / ch_v;

        const int query_head_strad = ch_qk;
        const int query_len_strad = n_head * query_head_strad;
        const int query_batch_strad = len_q * query_len_strad;

        const int key_head_strad = ch_qk;
        const int key_len_strad = n_head * key_head_strad;
        const int key_batch_strad = len_kv * key_len_strad;

        const int value_head_strad = ch_v;
        const int value_len_strad = n_head * value_head_strad;
        const int value_batch_strad = len_kv * value_len_strad;

        const int pos_head_strad = topk;
        const int pos_len_strad = n_head * pos_head_strad;
        const int pos_batch_strad = len_q * pos_len_strad;

        const int output_head_strad = ch_v;
        const int output_len_strad = n_head * output_head_strad;
        const int output_batch_strad = len_q * output_len_strad;

        const int attn_head_strad = topk;
        const int attn_len_strad = n_head * attn_head_strad;
        const int attn_batch_strad = len_q * attn_len_strad;

        const int grad_output_head_strad = ch_v;
        const int grad_output_len_strad = n_head * grad_output_head_strad;
        const int grad_output_batch_strad = len_q * grad_output_len_strad;

        // 首先计算 grad_value
        const int* pos_ptr = pos + batch_idx * pos_batch_strad + q_idx * pos_len_strad + head_idx * pos_head_strad; 
        const scalar_t* attn_ptr = attn + batch_idx * attn_batch_strad + q_idx * attn_len_strad + head_idx * attn_head_strad;
        const scalar_t* grad_output_ptr = grad_output + batch_idx * grad_output_batch_strad + \
                                          q_idx * grad_output_len_strad + head_idx * grad_output_head_strad;

        __shared__ scalar_t grad_attn[MAXN];
        __shared__ scalar_t grad_QK[MAXN];
        __shared__ scalar_t grad_sum[MAXN_HEAD];
        const int grad_attn_offset = thread_head_id * topk;

        if(thread_id == 0) grad_sum[thread_head_id] = 0;
        for(int i = thread_id ; i < topk ; i += ch_v) 
        {
            grad_attn[i + grad_attn_offset] = 0;
        }
        const int grad_QK_offset = thread_head_id * topk;
        for(int i = thread_id ; i < topk ; i += ch_v) 
        {
            grad_QK[i + grad_QK_offset] = 0;
        }

        for (int t = 0 ; t < topk ; ++ t)      
        {
            const int position = pos_ptr[t];
            scalar_t* grad_value_ptr = grad_value + batch_idx * value_batch_strad + position * value_len_strad + head_idx * value_head_strad; 
            const scalar_t* value_ptr = value + batch_idx * value_batch_strad + position * value_len_strad + head_idx * value_head_strad; 
            for (int j = thread_id ; j < ch_v ; j += ch_v)
            {
                atomicAdd(&grad_value_ptr[j], attn_ptr[t] * grad_output_ptr[j]);
                atomicAdd(&grad_attn[t + grad_attn_offset], value_ptr[j] * grad_output_ptr[j]);
            }
        }  
        __syncthreads();
        
        // 对于每个head 求 sum(attn[0~topk] * grad_attn[0~topk])
        for(int i = thread_id ; i < topk ; i += ch_v) 
        {
            atomicAdd(&grad_sum[thread_head_id], attn_ptr[i] * grad_attn[i + grad_attn_offset]);
        }       
        __syncthreads();

        // 求 grad_QK
        for(int i = thread_id ; i < topk ; i += ch_v) 
        {
            atomicAdd(&grad_QK[i + grad_QK_offset], attn_ptr[i] * (grad_attn[i + grad_attn_offset] - grad_sum[thread_head_id]));
        }
        __syncthreads();

        // 求 grad_query
        scalar_t* grad_query_ptr = grad_query + batch_idx * query_batch_strad + q_idx * query_len_strad + head_idx * query_head_strad;
        for (int t = 0 ; t < topk ; ++ t)      
        {
            const int position = pos_ptr[t];
            const scalar_t* key_ptr = key + batch_idx * key_batch_strad + position * key_len_strad + head_idx * key_head_strad;
            for (int j = thread_id ; j < ch_v ; j += ch_v)
            {
                atomicAdd(&grad_query_ptr[j], grad_QK[t + grad_QK_offset] * key_ptr[j] / fact);
            }
        } 
        __syncthreads();
        // 求grad_key
        const scalar_t* query_ptr = query + batch_idx * query_batch_strad + q_idx * query_len_strad + head_idx * query_head_strad;
        for (int t = 0 ; t < topk ; ++ t)      
        {
            const int position = pos_ptr[t];
            scalar_t* grad_key_ptr = grad_key + batch_idx * key_batch_strad + position * key_len_strad + head_idx * key_head_strad;
            for (int j = thread_id ; j < ch_v ; j += ch_v)
            {
                atomicAdd(&grad_key_ptr[j], grad_QK[t + grad_QK_offset] * query_ptr[j] / fact);
            }
        } 
    }
}


// template 关键字告诉C++编译器 下面是个泛型模板  
// 数据类型T 参数化数据类型
template <typename scalar_t>
void topk_attention_micro_batch_forward_cuda(cudaStream_t stream,
                              const scalar_t* query, 
                              const scalar_t* key,
                              const scalar_t* value,
                              const int* pos,
                              const int micro_batch, 
                              const int len_q, 
                              const int len_kv, 
                              const int n_head, 
                              const int ch_qk, 
                              const int ch_v, 
                              const int topk,
                              scalar_t* output,
                              scalar_t* attn) 
{

  const int num_kernels = micro_batch * len_q * n_head * ch_v;
  const int num_actual_kernels = micro_batch * len_q * n_head * ch_v;
  int num_threads = 0; 
  if(n_head * ch_v <= CUDA_NUM_THREADS && n_head * topk <= CUDA_NUM_THREADS) 
  {
    num_threads = n_head * ch_v; 
  } 
  else
  {
    num_threads = ch_v; 
  }

  topk_attention_micro_batch_forward_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
        num_kernels, query, key, value, pos, micro_batch, 
        len_q, len_kv, n_head, ch_qk, ch_v, topk, 
        output, attn
      );
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in topk_attention_micro_batch_forward_cuda: %s\n", cudaGetErrorString(err));
  }
}



template <typename scalar_t>
void topk_attention_micro_batch_backward_cuda(cudaStream_t stream,
                              const scalar_t* query, 
                              const scalar_t* key,
                              const scalar_t* value,
                              const int* pos, 
                              const scalar_t* attn,
                              const scalar_t* grad_output,
                              const int micro_batch, 
                              const int len_q, 
                              const int len_kv, 
                              const int n_head, 
                              const int ch_qk, 
                              const int ch_v, 
                              const int topk,
                              scalar_t* grad_query,
                              scalar_t* grad_key,
                              scalar_t* grad_value) 
{

  const int num_kernels = micro_batch * len_q * n_head * ch_v;
  const int num_actual_kernels = micro_batch * len_q * n_head * ch_v;
  int num_threads = 0; 
  if(n_head * ch_v <= CUDA_NUM_THREADS && n_head * topk <= CUDA_NUM_THREADS) 
  {
    num_threads = n_head * ch_v; 
  } 
  else
  {
    num_threads = ch_v; 
  }

  topk_attention_micro_batch_backward_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
        num_kernels, query, key, value, pos, attn, grad_output,micro_batch, 
        len_q, len_kv, n_head, ch_qk, ch_v, topk, 
        grad_query, grad_key, grad_value
      );
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in topk_attention_micro_batch_backward_cuda: %s\n", cudaGetErrorString(err));
  }
}

