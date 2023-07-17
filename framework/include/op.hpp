#ifndef __OP_HPP__
#define __OP_HPP__

#include "common.hpp"

namespace geefer
{

template<typename DType>
void vec_add(DType *input1, DType *input2, 
             DType *output, uint32_t size);


} // namespace geefer
#endif
