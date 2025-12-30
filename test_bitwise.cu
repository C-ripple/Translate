__global__ void bitwise(int *a) { int x = a[0]; a[0] = __popc(x) + __clz(x) + __brev(x) + __sad(x, 10, 0); }
