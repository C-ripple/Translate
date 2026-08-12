__global__ void butterflyReduce(float *data) {
    float val = data[threadIdx.x];
    for (int i = 1; i < 32; i *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    data[threadIdx.x] = val;
}
