/**
 * Test CUDA Kernels for CUDA to RIPPLE Translation
 * 
 * This file contains various CUDA patterns to test the translator:
 * - Basic kernels
 * - Shared memory usage
 * - Warp shuffles
 * - Atomics
 * - Reductions
 * - Multi-dimensional blocks
 */

#include <cuda_runtime.h>

// =============================================================================
// 1. Basic Vector Add - Simplest kernel pattern
// =============================================================================

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// =============================================================================
// 2. Vector Scale with Device Function
// =============================================================================

__device__ inline float scale_value(float x, float scale) {
    return x * scale;
}

__global__ void vector_scale(float *data, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = scale_value(data[idx], scale);
    }
}

// =============================================================================
// 3. Shared Memory Reduction
// =============================================================================

__global__ void reduce_sum(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// =============================================================================
// 4. Warp-Level Reduction with Shuffles
// =============================================================================

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void warp_reduce_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (idx < n) ? input[idx] : 0.0f;
    val = warp_reduce_sum(val);
    
    // First thread in each warp writes result
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(output, val);
    }
}

// =============================================================================
// 5. 2D Convolution with Shared Memory
// =============================================================================

#define TILE_SIZE 16
#define FILTER_SIZE 3

__global__ void conv2d(
    float *input, float *output, float *filter,
    int width, int height
) {
    __shared__ float tile[TILE_SIZE + FILTER_SIZE - 1][TILE_SIZE + FILTER_SIZE - 1];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    // Load tile with halo
    int halo = FILTER_SIZE / 2;
    int src_row = row - halo;
    int src_col = col - halo;
    
    if (src_row >= 0 && src_row < height && src_col >= 0 && src_col < width) {
        tile[ty][tx] = input[src_row * width + src_col];
    } else {
        tile[ty][tx] = 0.0f;
    }
    __syncthreads();
    
    // Compute convolution
    if (tx < TILE_SIZE && ty < TILE_SIZE && row < height && col < width) {
        float sum = 0.0f;
        for (int i = 0; i < FILTER_SIZE; i++) {
            for (int j = 0; j < FILTER_SIZE; j++) {
                sum += tile[ty + i][tx + j] * filter[i * FILTER_SIZE + j];
            }
        }
        output[row * width + col] = sum;
    }
}

// =============================================================================
// 6. Matrix Multiply (Simple)
// =============================================================================

__global__ void matmul(
    float *A, float *B, float *C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// =============================================================================
// 7. Histogram with Atomics
// =============================================================================

__global__ void histogram(
    unsigned char *data, int *histogram,
    int n, int num_bins
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        int bin = data[idx] * num_bins / 256;
        atomicAdd(&histogram[bin], 1);
    }
}

// =============================================================================
// 8. Prefix Sum (Inclusive Scan) - Warp Level
// =============================================================================

__global__ void prefix_sum_warp(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    
    float val = (idx < n) ? data[idx] : 0.0f;
    
    // Kogge-Stone scan within warp
    for (int d = 1; d < 32; d *= 2) {
        float temp = __shfl_up_sync(0xffffffff, val, d);
        if (lane >= d) {
            val += temp;
        }
    }
    
    if (idx < n) {
        data[idx] = val;
    }
}

// =============================================================================
// 9. Butterfly XOR Shuffle Pattern
// =============================================================================

__global__ void butterfly_exchange(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = data[idx];
        
        // Butterfly exchange pattern
        float partner = __shfl_xor_sync(0xffffffff, val, 1);
        val = val + partner;
        
        partner = __shfl_xor_sync(0xffffffff, val, 2);
        val = val + partner;
        
        partner = __shfl_xor_sync(0xffffffff, val, 4);
        val = val + partner;
        
        data[idx] = val;
    }
}

// =============================================================================
// 10. Softmax with Reduction
// =============================================================================

__global__ void softmax(float *input, float *output, int n) {
    __shared__ float smax;
    __shared__ float ssum;
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Find max (reduction)
    float local_max = (idx < n) ? input[idx] : -INFINITY;
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        float other = __shfl_down_sync(0xffffffff, local_max, s);
        local_max = fmaxf(local_max, other);
    }
    
    if (tid == 0) {
        smax = local_max;
    }
    __syncthreads();
    
    // Compute exp and sum
    float exp_val = (idx < n) ? expf(input[idx] - smax) : 0.0f;
    float local_sum = exp_val;
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, s);
    }
    
    if (tid == 0) {
        ssum = local_sum;
    }
    __syncthreads();
    
    // Normalize
    if (idx < n) {
        output[idx] = exp_val / ssum;
    }
}

// =============================================================================
// Host Launch Examples (for reference)
// =============================================================================

void launch_examples() {
    float *d_a, *d_b, *d_c;
    int n = 1024;
    
    // Allocate device memory
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    
    // Launch vector add
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vector_add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    
    // Launch 2D kernel
    dim3 block2d(16, 16);
    dim3 grid2d((n + 15) / 16, (n + 15) / 16);
    matmul<<<grid2d, block2d>>>(d_a, d_b, d_c, n, n, n);
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
