__global__ void test(float *a) { int i = blockIdx.x * blockDim.x + threadIdx.x; a[i] = 0; }
