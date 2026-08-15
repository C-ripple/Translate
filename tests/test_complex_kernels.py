"""
Complex CUDA Kernel Tests

Tests real-world kernel patterns to verify end-to-end translation quality.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from frontends.source.cuda_frontend import translate_cuda_source, CUDAToRIPPLETransformer
from core.semantic_model import TranslationContext, TranslationError


class TestMatrixOperations:
    """Tests for matrix multiplication and related operations."""
    
    def test_simple_matmul(self):
        """Test basic matrix multiplication kernel."""
        cuda_code = """
        __global__ void matmul(float *A, float *B, float *C, int N) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (row < N && col < N) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += A[row * N + k] * B[k * N + col];
                }
                C[row * N + col] = sum;
            }
        }
        """
        
        result = translate_cuda_source(cuda_code)
        
        # Verify key transformations
        assert "ripple_id(ripple_block, 0)" in result
        assert "ripple_id(ripple_block, 1)" in result
        assert "block_idx_x" in result
        assert "block_idx_y" in result
        assert "RIPPLE_SETUP_BLOCK()" in result
        assert "matmul_ripple" in result
    
    def test_tiled_matmul(self):
        """Test tiled matrix multiplication with shared memory.

        The two tiles are declared flat (1D) with manual row-major index
        arithmetic, not as CUDA's native tile_A[TILE_SIZE][TILE_SIZE] — a
        vtcm_malloc()-returned pointer can't be redeclared with a trailing
        [dim] the way a real 2D array can (see
        test_shared_memory_rule_multidim_hard_fails in test_translation.py),
        so a 2D __shared__ tile is a hard-fail and must be flattened by
        hand, exactly as the translator's own error message prescribes.
        This also exercises two __shared__ declarations in one kernel,
        i.e. the rule's right-to-left multi-match processing order.
        """
        cuda_code = """
        #define TILE_SIZE 16

        __global__ void matmul_tiled(float *A, float *B, float *C, int N) {
            __shared__ float tile_A[TILE_SIZE * TILE_SIZE];
            __shared__ float tile_B[TILE_SIZE * TILE_SIZE];

            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int row = blockIdx.y * TILE_SIZE + ty;
            int col = blockIdx.x * TILE_SIZE + tx;

            float sum = 0.0f;

            for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
                if (row < N && t * TILE_SIZE + tx < N)
                    tile_A[ty * TILE_SIZE + tx] = A[row * N + t * TILE_SIZE + tx];
                else
                    tile_A[ty * TILE_SIZE + tx] = 0.0f;

                if (col < N && t * TILE_SIZE + ty < N)
                    tile_B[ty * TILE_SIZE + tx] = B[(t * TILE_SIZE + ty) * N + col];
                else
                    tile_B[ty * TILE_SIZE + tx] = 0.0f;

                __syncthreads();

                for (int k = 0; k < TILE_SIZE; k++) {
                    sum += tile_A[ty * TILE_SIZE + k] * tile_B[k * TILE_SIZE + tx];
                }

                __syncthreads();
            }

            if (row < N && col < N) {
                C[row * N + col] = sum;
            }
        }
        """

        result = translate_cuda_source(cuda_code)

        # Verify shared memory translation to real VTCM malloc/free
        assert "vtcm_malloc(sizeof(float) * (TILE_SIZE * TILE_SIZE)" in result
        assert "vtcm_free(tile_A);" in result
        assert "vtcm_free(tile_B);" in result
        assert "ripple_id(ripple_block" in result
        # __syncthreads should be converted to comment (implicit in SIMD)
        assert "/* __syncthreads" in result or "__syncthreads" not in result


class TestReductionKernels:
    """Tests for reduction operations."""
    
    def test_warp_reduction_optimization(self):
        """Warp reduction loops are optimized to ripple_reduceadd, but a
        subsequent single-thread atomicAdd (writing the already-reduced
        value into global memory) correctly hard-fails — Ripple has no
        atomics API, and this translator doesn't yet model CUDA's
        multi-block grid execution, so there's no safe way to know whether
        a plain (non-atomic) write would be correct here."""
        cuda_code = """
        __global__ void reduce_sum(float *input, float *output, int N) {
            int tid = threadIdx.x;
            int idx = blockIdx.x * blockDim.x + tid;

            float val = (idx < N) ? input[idx] : 0.0f;

            // Warp reduction
            for (int offset = warpSize/2; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }

            if (tid == 0) {
                atomicAdd(output, val);
            }
        }
        """

        ctx = TranslationContext()
        transformer = CUDAToRIPPLETransformer(ctx)
        with pytest.raises(TranslationError):
            transformer.transform(cuda_code)

        # The shuffle loop itself is still correctly recognized — only the
        # trailing atomicAdd (no Ripple equivalent) is the hard-fail.
        assert any("Warp Reduction" in w for w in ctx.warnings)
        assert any("atomicAdd" in e for e in ctx.errors)
    
    def test_block_reduction(self):
        """Test block-level reduction with shared memory."""
        cuda_code = """
        __global__ void block_reduce(float *input, float *output) {
            __shared__ float shared[256];
            
            int tid = threadIdx.x;
            int idx = blockIdx.x * blockDim.x + tid;
            
            shared[tid] = input[idx];
            __syncthreads();
            
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] += shared[tid + s];
                }
                __syncthreads();
            }
            
            if (tid == 0) {
                output[blockIdx.x] = shared[0];
            }
        }
        """
        
        result = translate_cuda_source(cuda_code)
        
        assert "shared" in result
        assert "ripple_id(ripple_block, 0)" in result


class TestBitwiseOperations:
    """Tests for bitwise intrinsics."""
    
    def test_popcount(self):
        """Test population count intrinsic."""
        cuda_code = """
        __global__ void count_bits(int *input, int *output, int N) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < N) {
                output[idx] = __popc(input[idx]);
            }
        }
        """
        
        result = translate_cuda_source(cuda_code)
        assert "__builtin_popcount" in result
    
    def test_clz(self):
        """Test count leading zeros."""
        cuda_code = """
        __global__ void leading_zeros(int *input, int *output) {
            int idx = threadIdx.x;
            output[idx] = __clz(input[idx]);
        }
        """
        
        result = translate_cuda_source(cuda_code)
        assert "__builtin_clz" in result
    
    def test_bitreverse(self):
        """Test bit reversal."""
        cuda_code = """
        __global__ void reverse_bits(int *data) {
            int idx = threadIdx.x;
            data[idx] = __brev(data[idx]);
        }
        """
        
        result = translate_cuda_source(cuda_code)
        assert "__builtin_bitreverse32" in result


class TestAtomicOperations:
    """Tests for atomic operations."""
    
    def test_atomic_cas(self):
        """Ripple has no atomics API — atomicCAS must hard-fail, not
        translate to a fictional ripple_atomic_cas() call."""
        cuda_code = """
        __global__ void cas_kernel(int *lock) {
            int old = atomicCAS(lock, 0, 1);
        }
        """

        with pytest.raises(TranslationError):
            translate_cuda_source(cuda_code)

    def test_atomic_exch(self):
        """Ripple has no atomics API — atomicExch must hard-fail."""
        cuda_code = """
        __global__ void exch_kernel(int *data, int new_val) {
            int old = atomicExch(data, new_val);
        }
        """

        with pytest.raises(TranslationError):
            translate_cuda_source(cuda_code)

    def test_atomic_min_max(self):
        """Ripple has no atomics API — atomicMin/atomicMax must hard-fail."""
        cuda_code = """
        __global__ void minmax_kernel(int *data, int val) {
            atomicMin(data, val);
            atomicMax(data + 1, val);
        }
        """

        with pytest.raises(TranslationError):
            translate_cuda_source(cuda_code)


class TestConvolutionKernels:
    """Tests for convolution operations."""
    
    def test_1d_convolution(self):
        """Test 1D convolution kernel."""
        cuda_code = """
        __global__ void conv1d(float *input, float *kernel, float *output, 
                               int N, int K) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (idx < N) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    int pos = idx + k - K/2;
                    if (pos >= 0 && pos < N) {
                        sum += input[pos] * kernel[k];
                    }
                }
                output[idx] = sum;
            }
        }
        """
        
        result = translate_cuda_source(cuda_code)
        assert "ripple_id(ripple_block, 0)" in result
        assert "block_idx_x" in result
    
    def test_2d_convolution(self):
        """Test 2D convolution kernel (direct global-memory access — see
        test_shared_memory_rule_multidim_hard_fails in test_translation.py
        for the separate, dedicated 2D-shared-memory coverage)."""
        cuda_code = """
        #define KERNEL_SIZE 3

        __global__ void conv2d(float *input, float *kernel, float *output,
                               int width, int height) {
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int x = blockIdx.x * blockDim.x + tx;
            int y = blockIdx.y * blockDim.y + ty;

            if (x < width && y < height) {
                float sum = 0.0f;
                for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                    for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                        int sx = x + kx - KERNEL_SIZE / 2;
                        int sy = y + ky - KERNEL_SIZE / 2;
                        if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                            sum += input[sy * width + sx] *
                                   kernel[ky * KERNEL_SIZE + kx];
                        }
                    }
                }
                output[y * width + x] = sum;
            }
        }
        """

        result = translate_cuda_source(cuda_code)
        assert "ripple_id(ripple_block, 0)" in result
        assert "ripple_id(ripple_block, 1)" in result

    def test_shared_memory_vtcm_free_placement(self):
        """vtcm_free() must be inserted at the kernel's closing brace, after
        the nested for/if statements that use the shared buffer — not
        immediately after the malloc."""
        cuda_code = """
        __global__ void reduce_sum(float *input, float *output, int n) {
            __shared__ float sdata[256];
            int tid = threadIdx.x;
            sdata[tid] = input[tid];
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
            }
            if (tid == 0) output[0] = sdata[0];
        }
        """

        result = translate_cuda_source(cuda_code)
        assert "vtcm_malloc(sizeof(float) * (256)" in result
        assert "vtcm_free(sdata);" in result
        assert result.index("for (") < result.index("vtcm_free(sdata)")


class TestImageProcessing:
    """Tests for image processing kernels."""
    
    def test_rgb_to_grayscale(self):
        """Test RGB to grayscale conversion."""
        cuda_code = """
        __global__ void rgb_to_gray(unsigned char *rgb, unsigned char *gray, 
                                    int width, int height) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (x < width && y < height) {
                int idx = y * width + x;
                int rgb_idx = idx * 3;
                
                unsigned char r = rgb[rgb_idx];
                unsigned char g = rgb[rgb_idx + 1];
                unsigned char b = rgb[rgb_idx + 2];
                
                gray[idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
            }
        }
        """
        
        result = translate_cuda_source(cuda_code)
        assert "ripple_id(ripple_block" in result
        assert "block_idx_x" in result and "block_idx_y" in result
    
    def test_sad_rename(self):
        """__sad has no real Ripple equivalent — it translates to the
        translator's own cripple_sad helper, not a ripple_-prefixed name
        that would imply a real API symbol."""
        cuda_code = """
        __global__ void compute_sad_only(unsigned char *img1, unsigned char *img2,
                                         int *out, int N) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < N) {
                out[idx] = __sad(img1[idx], img2[idx], 0);
            }
        }
        """

        result = translate_cuda_source(cuda_code)
        assert "cripple_sad(" in result
        # Pin the call site and the #define together — nothing else in the
        # suite exercises __sad, so a future edit renaming only one side
        # would otherwise pass this test while calling an undefined macro.
        assert "#define cripple_sad" in result

    def test_sad_with_bare_atomic_add_fails(self):
        """A bare per-thread atomicAdd (not a recognized block-reduction
        idiom) has no Ripple equivalent, independent of the __sad rename."""
        cuda_code = """
        __global__ void compute_sad(unsigned char *img1, unsigned char *img2,
                                    int *sad, int N) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < N) {
                int local_sad = __sad(img1[idx], img2[idx], 0);
                atomicAdd(sad, local_sad);
            }
        }
        """

        with pytest.raises(TranslationError):
            translate_cuda_source(cuda_code)


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_kernel(self):
        """Test empty kernel translation."""
        cuda_code = "__global__ void empty() {}"
        result = translate_cuda_source(cuda_code)
        assert "empty_ripple" in result
    
    def test_no_thread_idx(self):
        """Test kernel without thread indexing."""
        cuda_code = """
        __global__ void constant_fill(float *data, float value) {
            data[0] = value;
        }
        """
        result = translate_cuda_source(cuda_code)
        assert "constant_fill_ripple" in result
    
    def test_multiple_kernels(self):
        """Test file with multiple kernel definitions."""
        cuda_code = """
        __global__ void kernel1(int *a) { a[threadIdx.x] = 1; }
        __global__ void kernel2(int *b) { b[threadIdx.x] = 2; }
        """
        result = translate_cuda_source(cuda_code)
        assert "kernel1_ripple" in result
        assert "kernel2_ripple" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
