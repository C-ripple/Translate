__global__ void butterflyXor(int *data) {
    int val = data[threadIdx.x];
    int partner_val = __shfl_xor_sync(0xffffffff, val, 1);
    data[threadIdx.x] = val + partner_val;
}
