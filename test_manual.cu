__global__ void test(float *a) { int i = threadIdx.x; a[i] = 0; }
