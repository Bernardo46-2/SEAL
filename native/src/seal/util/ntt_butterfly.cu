#include "seal/util/ntt_butterfly_cuda.h"
#include <cuda_runtime.h>

struct Root64 {
    uint64_t operand;
    uint64_t quotient;
};

__device__ __forceinline__ uint64_t add_mod(uint64_t x, uint64_t y, uint64_t mod) {
    return x + y;
}

__device__ __forceinline__ uint64_t sub_mod(uint64_t x, uint64_t y, uint64_t mod) {
    return x + (mod << 1) - y;
}

__device__ __forceinline__ uint64_t mul_root(uint64_t x, Root64 y, uint64_t mod) {
    uint64_t tmp1 = __umul64hi(x, y.quotient);
    uint64_t tmp2 = y.operand * x - tmp1 * mod;
    uint64_t mod_times_two = mod << 1;
    return tmp2 >= mod_times_two ? tmp2 - mod_times_two : tmp2;
}

__device__ __forceinline__ uint64_t guard(uint64_t x, uint64_t mod) {
    uint64_t mod_times_two = mod << 1;
    return x >= mod_times_two ? x - mod_times_two : x;
}

__global__ void transform_to_rev_kernel_1(
    uint64_t* values, 
    int log_n, 
    const Root64* roots, 
    uint64_t modulus, 
    int m,
    int gap) 
{
    size_t n = 1ULL << log_n;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    size_t total_threads = n >> 1;
    if(tid >= total_threads) return;

    // Map thread → (i, j)
    size_t group = tid / gap;   // i
    size_t j = tid % gap;

    if(group >= m) return;

    size_t offset = group * (gap << 1);

    size_t idx_x = offset + j;
    size_t idx_y = idx_x + gap;

    Root64 r = roots[group];

    uint64_t u = guard(values[idx_x], modulus);
    uint64_t v = mul_root(values[idx_y], r, modulus);

    values[idx_x] = add_mod(u, v, modulus);
    values[idx_y] = sub_mod(u, v, modulus);
}

__global__ void transform_to_rev_kernel_2(
    uint64_t* values,
    const Root64* roots,
    uint64_t modulus,
    int m)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= m) return;

    int idx_x = 2 * tid;
    int idx_y = idx_x + 1;

    Root64 r = roots[tid];

    uint64_t u = guard(values[idx_x], modulus);
    uint64_t v = mul_root(values[idx_y], r, modulus);
    
    values[idx_x] = add_mod(u, v, modulus);
    values[idx_y] = sub_mod(u, v, modulus);
}

void transform_to_rev_cuda(
    uint64_t* values, 
    int log_n, 
    const seal::util::MultiplyUIntModOperand* roots, 
    uint64_t modulus) 
{
    size_t n = size_t(1) << log_n;
    
    // Device memory
    uint64_t* d_values;
    Root64* d_roots;

    std::vector<Root64> stage_roots;
    const seal::util::MultiplyUIntModOperand* root_ptr = roots;
    int gap = n >> 1;
    int m = 1;

    for(; m < (n >> 1); m <<= 1) {
        for(int i = 0; i < m; i++) {
            root_ptr++;

            Root64 r;
            r.operand = root_ptr->operand;
            r.quotient = root_ptr->quotient;
            
            stage_roots.push_back(r);
        }
        gap >>= 1;
    }

    for(int i = 0; i < m; i++) {
        root_ptr++;

        Root64 r;
        r.operand = root_ptr->operand;
        r.quotient = root_ptr->quotient;

        stage_roots.push_back(r);
    }

    cudaMalloc(&d_values, n * sizeof(uint64_t));
    cudaMemcpy(d_values, values, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_roots, stage_roots.size() * sizeof(Root64));
    cudaMemcpy(d_roots, stage_roots.data(),
               stage_roots.size() * sizeof(Root64),
               cudaMemcpyHostToDevice);

    // --- Launch kernels ---
    int total_threads = n >> 1;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    gap = n >> 1;
    m = 1;
    int root_offset = 0;

    for(; m < (n >> 1); m <<= 1) {
        transform_to_rev_kernel_1<<<blocks, threads_per_block>>>(
            d_values,
            log_n,
            d_roots + root_offset,
            modulus,
            m,
            gap
        );

        cudaDeviceSynchronize();

        root_offset += m;
        gap >>= 1;
    }
    
    transform_to_rev_kernel_2<<<blocks, threads_per_block>>>(
        d_values,
        d_roots + root_offset,
        modulus,
        m
    );
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(values, d_values, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaFree(d_roots);
}
