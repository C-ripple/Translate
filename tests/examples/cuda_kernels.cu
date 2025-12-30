/**
 * CUDA Example Kernels for Testing CUDA-to-RIPPLE Translation
 * 
 * This file contains various CUDA patterns that should be translated
 * to their RIPPLE equivalents for Hexagon HVX.
 */

#include <cuda_runtime.h>
#include <stdio.h>

// =============================================================================
// Example 1: Simple Vector Addition (Element-wise Operation)
// =============================================================================

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// =============================================================================
// Example 2: Vector Scaling with Device Function
// =============================================================================

__device__ inline float scale(float x, float factor) {
    return x * factor;
}

__global__ void vectorScale(float *data, float factor, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] = scale(data[idx], factor);
    }
}

// =============================================================================
// Example 3: Parallel Reduction with Shared Memory
// =============================================================================

__global__ void reduceSum(float *input, float *output, int n) {
    __shared__ float sdata[256];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// =============================================================================
// Example 4: Matrix Transpose (2D Grid)
// =============================================================================

#define TILE_DIM 32

__global__ void transpose(float *odata, const float *idata, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts
    
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    
    int index_in = xIndex + width * yIndex;
    
    // Load tile into shared memory
    if (xIndex < width && yIndex < height) {
        tile[threadIdx.y][threadIdx.x] = idata[index_in];
    }
    __syncthreads();
    
    // Write transposed tile
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + height * yIndex;
    
    if (xIndex < height && yIndex < width) {
        odata[index_out] = tile[threadIdx.x][threadIdx.y];
    }
}

// =============================================================================
// Example 5: Warp Shuffle Reduction
// =============================================================================

__device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduceWarpShuffle(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // Warp-level reduction
    val = warpReduceSum(val);
    
    // First thread in each warp writes result
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(output, val);
    }
}

// =============================================================================
// Example 6: Histogram with Atomics
// =============================================================================

__global__ void histogram(const unsigned char *data, int *hist, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        unsigned char bin = data[idx];
        atomicAdd(&hist[bin], 1);
    }
}

// =============================================================================
// Example 7: Stencil Operation (1D convolution)
// =============================================================================

#define RADIUS 3

__global__ void stencil1D(const float *input, float *output, int n) {
    __shared__ float temp[256 + 2 * RADIUS];
    
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;
    
    // Load center elements
    if (gindex < n) {
        temp[lindex] = input[gindex];
    }
    
    // Load halo elements
    if (threadIdx.x < RADIUS) {
        int halo_left = gindex - RADIUS;
        int halo_right = gindex + blockDim.x;
        
        temp[lindex - RADIUS] = (halo_left >= 0) ? input[halo_left] : 0.0f;
        temp[lindex + blockDim.x] = (halo_right < n) ? input[halo_right] : 0.0f;
    }
    __syncthreads();
    
    // Apply stencil
    if (gindex < n) {
        float result = 0.0f;
        for (int offset = -RADIUS; offset <= RADIUS; offset++) {
            result += temp[lindex + offset];
        }
        output[gindex] = result / (2 * RADIUS + 1);
    }
}

// =============================================================================
// Example 8: Dot Product with Multiple Reductions
// =============================================================================

__global__ void dotProduct(const float *a, const float *b, float *result, int n) {
    __shared__ float cache[256];
    
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Compute partial products
    float temp = 0.0f;
    while (idx < n) {
        temp += a[idx] * b[idx];
        idx += blockDim.x * gridDim.x;  // Grid-stride loop
    }
    
    cache[tid] = temp;
    __syncthreads();
    
    // Reduction
    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (tid < i) {
            cache[tid] += cache[tid + i];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, cache[0]);
    }
}

// =============================================================================
// Example 9: Prefix Sum (Scan)
// =============================================================================

__global__ void inclusiveScan(float *data, int n) {
    __shared__ float temp[256];
    
    int tid = threadIdx.x;
    int pout = 0, pin = 1;
    
    // Load into shared memory
    temp[tid] = (tid < n) ? data[tid] : 0;
    __syncthreads();
    
    // Up-sweep (reduce) phase
    for (int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pin;
        
        if (tid >= offset) {
            temp[pout * n + tid] = temp[pin * n + tid] + temp[pin * n + tid - offset];
        } else {
            temp[pout * n + tid] = temp[pin * n + tid];
        }
        __syncthreads();
    }
    
    data[tid] = temp[pout * n + tid];
}

// =============================================================================
// Example 10: Softmax (Numerically Stable)
// =============================================================================

__global__ void softmax(float *input, float *output, int n) {
    __shared__ float shared_max;
    __shared__ float shared_sum;
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Find max (for numerical stability)
    float local_max = (idx < n) ? input[idx] : -INFINITY;
    
    // Warp reduce to find max
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = fmaxf(local_max, other);
    }
    
    if (tid % warpSize == 0) {
        atomicMax((int*)&shared_max, __float_as_int(local_max));
    }
    __syncthreads();
    
    // Compute exp(x - max)
    float val = (idx < n) ? expf(input[idx] - shared_max) : 0.0f;
    
    // Sum reduction
    float local_sum = val;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    if (tid % warpSize == 0) {
        atomicAdd(&shared_sum, local_sum);
    }
    __syncthreads();
    
    // Normalize
    if (idx < n) {
        output[idx] = val / shared_sum;
    }
}

// =============================================================================
// Host Code for Testing
// =============================================================================

int main() {
    printf("CUDA Example Kernels - Ready for RIPPLE Translation\n");
    printf("These kernels demonstrate various CUDA patterns:\n");
    printf("  1. Element-wise operations\n");
    printf("  2. Device functions\n");
    printf("  3. Shared memory reductions\n");
    printf("  4. 2D operations (transpose)\n");
    printf("  5. Warp shuffle primitives\n");
    printf("  6. Atomic operations\n");
    printf("  7. Stencil/convolution\n");
    printf("  8. Grid-stride loops\n");
    printf("  9. Prefix sum (scan)\n");
    printf(" 10. Complex patterns (softmax)\n");
    
    return 0;
}
