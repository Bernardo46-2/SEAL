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

__device__ __forceinline__ uint64_t mul_mod(uint64_t x, Root64 y, uint64_t mod) {
    uint64_t tmp = __umul64hi(x, y.quotient);
    return x * y.operand - tmp * mod;
}

__device__ __forceinline__ uint64_t guard(uint64_t &x, uint64_t mod) {
    uint64_t mod_x_2 = mod << 1;
    return x >= mod_x_2 ? x - mod_x_2 : x;
}

__global__ void ntt_stage_kernel(
    uint64_t* values, 
    int log_n, 
    const Root64* roots, 
    uint64_t modulus, 
    int stage,
    int m,
    int gap) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = 1 << log_n;
    int butterflies = n >> 1;

    if (tid >= butterflies) return;

    int group = tid / gap;
    int j = tid % gap;

    uint64_t x_idx = group * (gap << 1) + j;
    uint64_t y_idx = x_idx + gap;

    Root64 r = roots[group];

    uint64_t u = values[x_idx];
    uint64_t v = mul_mod(values[y_idx], r, modulus);

    values[x_idx] = add_mod(u, v, modulus);
    values[y_idx] = sub_mod(u, v, modulus);
}

void transform_to_rev_cuda(
    uint64_t* values, 
    int log_n, 
    const seal::util::MultiplyUIntModOperand* roots, 
    uint64_t modulus) 
{
    size_t n = size_t(1) << log_n;
    size_t butterflies = n >> 1;

    // Device memory
    uint64_t* d_values;
    Root64* d_roots;

    cudaMalloc(&d_values, n * sizeof(uint64_t));
    cudaMemcpy(d_values, values, n * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // --- Precompute stage roots ---
    std::vector<Root64> stage_roots;

    int gap = n >> 1;
    int m = 1;

    const seal::util::MultiplyUIntModOperand* root_ptr = roots;

    for (; m < (n >> 1); m <<= 1)
    {
        for (int i = 0; i < m; i++)
        {
            root_ptr++;

            Root64 r;
            r.operand = root_ptr->operand;
            r.quotient = root_ptr->quotient;
            
            stage_roots.push_back(r);
        }
        gap >>= 1;
    }

    cudaMalloc(&d_roots, stage_roots.size() * sizeof(Root64));
    cudaMemcpy(d_roots, stage_roots.data(),
               stage_roots.size() * sizeof(Root64),
               cudaMemcpyHostToDevice);

    // --- Launch kernels ---
    int threads = 256;
    int blocks = (butterflies + threads - 1) / threads;

    int stage = 0;
    gap = n >> 1;
    m = 1;
    int root_offset = 0;

    for (; m < (n >> 1); m <<= 1, stage++)
    {
        ntt_stage_kernel<<<blocks, threads>>>(
            d_values,
            log_n,
            d_roots + root_offset,
            modulus,
            stage,
            m,
            gap
        );

        cudaDeviceSynchronize();

        root_offset += m;
        gap >>= 1;
    }

    // Copy back
    cudaMemcpy(values, d_values, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaFree(d_roots);
}
