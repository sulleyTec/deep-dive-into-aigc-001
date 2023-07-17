#include "op.hpp"


namespace geefer {

template<typename DType>
__global__ void vec_add_kernel(DType *input1, 
                               DType *input2, 
                               DType *output, 
                               uint32_t size)
{
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if (idx < size)
        output[idx] = input1[idx] + input2[idx];
}

template<typename DType>
void vec_add(DType *input1, DType *input2, 
             DType *output, uint32_t size) {

    int block_size = 256;
    int grid = (size+block_size-1)/block_size;

    vec_add_kernel<<<grid, block_size>>>(input1, input2, output, size);
}

} // namespace geefer



