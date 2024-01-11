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
        // printf("The index is %d\n", index);
        int idx = index;
        const int ch_idx = idx % ch_v;
        idx /= ch_v;
        const int head_idx = idx % n_head;
        idx /= n_head;
        const int q_idx = idx % len_q;
        idx /= len_q;
        const int batch_idx = idx; 
        // 事实上 ch_idx 应当等于 thread_id
        int thread_id = threadIdx.x;
        // printf("The thread_id is %d, and the ch_idx is %d\n", thread_id, ch_idx);
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

        __shared__ scalar_t temp_buffer[MAXN];
        if(thread_id < topk) {
            temp_buffer[thread_id] = 0;
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
            //   if(q_idx == 0) 
            //   {
            //         printf("batch_idx is %d, q_idx is %d, head_idx is %d, ch_idx is %d\n  \
            //             t is %d, pos is %d, query is %lf, key is %lf\n",  \
            //             batch_idx, q_idx, head_idx, ch_idx,     \
            //             t, position, query_ptr[j], key_ptr[j]);
            //   }
            } 
            atomicAdd(&temp_buffer[t], thread_sum);
        }
        __syncthreads();

        // if(thread_id == 0 && q_idx == 0) {
        //     for (int t = 0 ; t < topk ; ++ t) 
        //     {
        //         printf("temp_buffer[%d] is %lf ", t, temp_buffer[t]);
        //     }
        //     printf("\n");
        // }        

        __shared__ scalar_t sh_maxv, sh_sum;
        if (threadIdx.x == 0) {
            sh_maxv = -FLT_MAX;
            sh_sum = 0;
        }

        if(thread_id < topk) {
            temp_buffer[thread_id] /= fact;
            sh_maxv = max(sh_maxv, temp_buffer[thread_id]);
        }
        __syncthreads();

        if(thread_id < topk) {
            temp_buffer[thread_id] = exp(temp_buffer[thread_id] - sh_maxv);
            atomicAdd(&sh_sum, temp_buffer[thread_id]);
        }
        __syncthreads();


        if(thread_id < topk) {
            temp_buffer[thread_id] /= sh_sum;
        }
        __syncthreads();

        scalar_t* output_ptr = output + batch_idx * output_batch_strad + q_idx * output_len_strad + head_idx * output_head_strad;
        for (int t = 0 ; t < topk ; ++ t) 
        {
            const int position = pos_ptr[t];
            const scalar_t* value_ptr = value + batch_idx * value_batch_strad + position * value_len_strad + head_idx * value_head_strad;  
            for (int j = thread_id ; j < ch_v ; j += ch_v)
            {
                atomicAdd(&output_ptr[j], value_ptr[j] * temp_buffer[t]);
            }
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
                              scalar_t* output) 
{

  const int num_kernels = micro_batch * len_q * n_head * ch_v;
  const int num_actual_kernels = micro_batch * len_q * n_head * ch_v;
  const int num_threads = ch_v; 

/*
    inline int GET_BLOCKS(const int N, const int num_threads)
    {
        return (N + num_threads - 1) / num_threads;
    }
    核函数只能在主机端调用，调用时必须申明执行参数。调用形式如下：
    Kernel<<<Dg, Db, Ns, S>>>(paramlist);
    •  <<< >>>内是核函数的执行参数，告诉编译器运行时如何启动核函数，用于说明内核函数中的线程数量，以及线程是如何组织的。
    •  参数Dg用于定义整个grid的维度和尺寸，即一个grid有多少个block, 为dim3类型
        Dim3 Dg(Dg.x, Dg.y, 1)表示grid中每行有Dg.x个block，每列有Dg.y个block，第三维一般为1.
        (目前一个核函数对应一个grid), 这样整个grid中共有Dg.x*Dg.y个block。
    •  参数Db用于定义一个block的维度和尺寸，即一个block有多少个thread，为dim3类型。
        Dim3 Db(Db.x, Db.y, Db.z)表示整个block中每行有Db.x个thread，每列有Db.y个thread，高度为Db.z。
        Db.x 和 Db.y最大值为1024，Db.z最大值为64。一个block中共有Db.x*Db.y*Db.z个thread。
    •  参数Ns是一个可选参数，用于设置每个block除了静态分配的shared Memory以外，最多能动态分配的shared memory
        大小，单位为byte。不需要动态分配时该值为0或省略不写。
    •  参数S是一个cudaStream_t类型的可选参数，初始值为零，表示该核函数处在哪个流之中。
*/

  // 这里相当于每个block的线程数和 ch_v 相等
  topk_attention_micro_batch_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
        num_kernels, query, key, value, pos, micro_batch, 
        len_q, len_kv, n_head, ch_qk, ch_v, topk, 
        output
      );
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in topk_attention_micro_batch_cuda: %s\n", cudaGetErrorString(err));
  }
}



