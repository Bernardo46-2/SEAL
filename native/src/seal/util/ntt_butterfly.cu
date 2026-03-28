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

__device__ uint64_t compute_quotient(uint64_t operand, uint64_t modulus)
{
    uint64_t hi = operand;
    uint64_t lo = 0;

    uint64_t quotient = 0;
    uint64_t remainder = 0;

    for (int i = 127; i >= 0; i--) {
        remainder <<= 1;

        if (i >= 64)
            remainder |= (hi >> (i - 64)) & 1ULL;
        else
            remainder |= (lo >> i) & 1ULL;

        if (remainder >= modulus) {
            remainder -= modulus;
            if (i < 64)
                quotient |= (1ULL << i);
        }
    }

    return quotient;
}

__device__ __forceinline__ Root64 mul_root_scalar(Root64 r, Root64 s, uint64_t mod) {
    uint64_t tmp1 = __umul64hi(r.operand, s.quotient);
    uint64_t tmp2 = s.operand * r.operand - tmp1 * mod;
    uint64_t operand = tmp2 >= mod ? tmp2 - mod : tmp2;

    return (Root64) { .operand = operand, .quotient = compute_quotient(operand, mod) };
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

    size_t group = tid / gap;
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

    cudaMemcpy(values, d_values, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaFree(d_roots);
}

__global__ void transform_from_rev_kernel_1(
    uint64_t* values, 
    int log_n, 
    const Root64* roots, 
    uint64_t modulus, 
    int m,
    int gap) 
{
    size_t n = 1ULL << log_n;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = n >> 1;

    if(tid >= total_threads) return;
    
    size_t group = tid / gap;
    size_t j = tid % gap;

    if(group >= m) return;

    size_t offset = group * (gap << 1);

    size_t idx_x = offset + j;
    size_t idx_y = idx_x + gap;

    Root64 r = roots[group];

    uint64_t u = values[idx_x];
    uint64_t v = values[idx_y];

    uint64_t sum = add_mod(u, v, modulus);
    uint64_t diff = sub_mod(u, v, modulus);

    values[idx_x] = guard(sum, modulus);
    values[idx_y] = mul_root(diff, r, modulus);
}

__global__ void transform_from_rev_kernel_2_no_scalar(
    uint64_t* values,
    Root64* roots,
    uint64_t modulus,
    int gap)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= gap) return;

    Root64 r = roots[0];

    int idx_x = tid;
    int idx_y = tid + gap;

    uint64_t u = values[idx_x];
    uint64_t v = values[idx_y];

    uint64_t sum = add_mod(u, v, modulus);
    uint64_t diff = sub_mod(u, v, modulus);

    values[idx_x] = guard(sum, modulus);
    values[idx_y] = mul_root(diff, r, modulus);
}

__global__ void transform_from_rev_kernel_2_with_scalar(
    uint64_t* values,
    Root64* roots,
    uint64_t modulus,
    int gap,
    Root64 scalar)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= gap) return;

    Root64 r = roots[0];
    Root64 scaled_r = mul_root_scalar(r, scalar, modulus);
    
    int idx_x = tid;
    int idx_y = tid + gap;

    uint64_t u = guard(values[idx_x], modulus);
    uint64_t v = values[idx_y];

    uint64_t sum = add_mod(u, v, modulus);
    uint64_t diff = sub_mod(u, v, modulus);

    values[idx_x] = mul_root(guard(sum, modulus), scalar, modulus);
    values[idx_y] = mul_root(diff, scaled_r, modulus);
}

void transform_from_rev_cuda(
    uint64_t* values, 
    int log_n, 
    const seal::util::MultiplyUIntModOperand* roots, 
    uint64_t modulus,
    const seal::util::MultiplyUIntModOperand* scalar) 
{
    size_t n = size_t(1) << log_n;
    
    uint64_t* d_values;
    Root64* d_roots;

    std::vector<Root64> stage_roots;
    const seal::util::MultiplyUIntModOperand* root_ptr = roots;
    int m = n >> 1;
    int gap = 1;

    for(; m > 1; m >>= 1) {
        for(std::size_t i = 0; i < m; i++) {
            root_ptr++;

            Root64 r;
            r.operand = root_ptr->operand;
            r.quotient = root_ptr->quotient;
            
            stage_roots.push_back(r);
        }
    }

    root_ptr++;

    Root64 r;
    r.operand = root_ptr->operand;
    r.quotient = root_ptr->quotient;

    stage_roots.push_back(r);
    
    cudaMalloc(&d_values, n * sizeof(uint64_t));
    cudaMemcpy(d_values, values, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_roots, stage_roots.size() * sizeof(Root64));
    cudaMemcpy(d_roots, stage_roots.data(),
               stage_roots.size() * sizeof(Root64),
               cudaMemcpyHostToDevice);
    
    int total_threads = n >> 1;
    int threads_per_block = 256;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    gap = 1;
    m = n >> 1;
    int root_offset = 0;

    for(; m > 1; m >>= 1) {
        transform_from_rev_kernel_1<<<blocks, threads_per_block>>>(
            d_values,
            log_n,
            d_roots + root_offset,
            modulus,
            m,
            gap
        );

        cudaDeviceSynchronize();

        root_offset += m;
        gap <<= 1;
    }
    
    total_threads = gap;
    blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    if(scalar != nullptr) {
        Root64 s;
        s.operand = scalar->operand;
        s.quotient = scalar->quotient;

        transform_from_rev_kernel_2_with_scalar<<<blocks, threads_per_block>>>(
            d_values, 
            d_roots + root_offset, 
            modulus, 
            gap,
            s
        );
        cudaDeviceSynchronize();
    } else {
        transform_from_rev_kernel_2_no_scalar<<<blocks, threads_per_block>>>(
            d_values, 
            d_roots + root_offset, 
            modulus, 
            gap
        );
        cudaDeviceSynchronize();
    }

    cudaMemcpy(values, d_values, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaFree(d_roots);
}
