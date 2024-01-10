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
const int MAXN = 100;
inline int GET_BLOCKS(const int N, const int num_threads)
{
  return (N + num_threads - 1) / num_threads;
}



template <typename scalar_t>
__global__ void topk_attention_micro_batch_gpu_kernel(  const int n, // n = micro_batch * len_q * n_head * ch_v = element(output)
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
                                                        scalar_t* temp,   // temp   (micro_batch, len_q, n_head, topk)
                                                        scalar_t* output) // output (micro_batch, len_q, n_head, ch_v)
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
        const int temp_head_strad = topk;
        const int temp_len_strad = n_head * temp_head_strad;
        const int temp_batch_strad = len_q * temp_len_strad;
        const int output_head_strad = ch_v;
        const int output_len_strad = n_head * output_head_strad;
        const int output_batch_strad = len_q * output_len_strad;

        const int* pos_ptr = pos + batch_idx * pos_batch_strad + q_idx * pos_len_strad + head_idx * pos_head_strad; 
        scalar_t* temp_ptr = temp + batch_idx * temp_batch_strad + q_idx * temp_len_strad + head_idx * temp_head_strad;
        const scalar_t* query_ptr = query + batch_idx * query_batch_strad + q_idx * query_len_strad + head_idx * query_head_strad;
        for (int t = 0 ; t < topk ; ++ t) 
        {
            const int position = pos_ptr[t];
            const scalar_t* key_ptr = key + batch_idx * key_batch_strad + position * key_len_strad + n_head * key_head_strad;
            for(int j = 0 ; j < ch_qk ; j += ch_v)
            {
              temp_ptr[t] += query_ptr[j] * key_ptr[j];
            } 
        }
        __syncthreads();


        __shared__ scalar_t maxv_buffer[MAXN];
        __shared__ scalar_t sum_buffer[MAXN];
        int thread_id = blockDim.x;
        if(thread_id < topk) {
            maxv_buffer[thread_id] = temp_ptr[thread_id];
        }
        __syncthreads();
        

        if(thread_id < topk) {
            maxv_buffer[0] = max(maxv_buffer[0], maxv_buffer[thread_id]);
        }
        __syncthreads();


        if(thread_id < topk) {
            temp_ptr[thread_id] = exp(temp_ptr[thread_id] - maxv_buffer[0]);
            sum_buffer[0] += temp_ptr[thread_id];
        }
        __syncthreads();


        if(thread_id < topk) {
            temp_ptr[thread_id] /= sum_buffer[0];
        }
        __syncthreads();

        scalar_t* output_ptr = output + batch_idx * output_batch_strad + q_idx * output_len_strad + head_idx * output_head_strad;
        for (int t = 0 ; t < topk ; ++ t) 
        {
            const int position = pos_ptr[t];
            const scalar_t* value_ptr = value + batch_idx * value_batch_strad + position * value_len_strad + n_head * value_head_strad;  
            output_ptr[thread_id] += value[thread_id] * temp_ptr[t];
        }  
    }
}




// template 关键字告诉C++编译器 下面是个泛型模板  
// 数据类型T 参数化数据类型
template <typename scalar_t>
void topk_attention_micro_batch_cuda(cudaStream_t stream,
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
                              scalar_t* temp,
                              scalar_t* output) 
{

  const int num_kernels = micro_batch * len_q * n_head * ch_v;
  const int num_actual_kernels = micro_batch * len_q * n_head * ch_v;
  const int num_threads = ch_v; 

  topk_attention_micro_batch_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
        num_kernels, query, key, value, pos, micro_batch, 
        len_q, len_kv, n_head, ch_qk, ch_v, topk, 
        temp, output
      );
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in topk_attention_micro_batch_cuda: %s\n", cudaGetErrorString(err));
  }
}



