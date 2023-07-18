#ifndef __OP_HPP__
#define __OP_HPP__

#include "common.hpp"

namespace geefer
{

template<typename DType>
__global__ void vec_add_kernel(const DType *input1, 
                               const DType *input2, 
                               DType *output, 
                               const uint32_t size);

template<typename DType>
void vec_add(const DType *input1, const DType *input2, 
             DType *output, const uint32_t size);


} // namespace geefer
#endif
