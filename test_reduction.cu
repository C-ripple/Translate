__global__ void reduce(float *val) { 
 float sum = *val; 
 for (int offset = warpSize/2; offset > 0; offset /= 2) { 
 sum += __shfl_down_sync(0xff, sum, offset); 
 } 
 *val = sum; 
 }
