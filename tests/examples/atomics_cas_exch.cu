__global__ void test_atomics(int *a, int *b) { 
 int old = atomicCAS(a, 0, 1); 
 int exc = atomicExch(b, 2); 
 }
