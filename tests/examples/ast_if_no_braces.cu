__global__ void vectorAdd(float *A, float *B, float *C, int N) { 
 int i = threadIdx.x; 
 if (i < N) C[i] = A[i] + B[i]; 
 }
