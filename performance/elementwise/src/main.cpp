#include <iostream>
#include <cuda_fp16.h>
#include "elementwise.hpp"

int main() {

    const int64_t num_elements = 163840*1000; 
    int grid_dim = 501;
    int iters = 10;

    std::cout << "elementwise_mul_half_naive:" << std::endl;
    elementwise_mul_half_naive(num_elements, iters, grid_dim);
    std::cout << std::endl;

    std::cout << "elementwise_mul_half_optimized:" << std::endl;
    elementwise_mul_half_optimized(num_elements, iters);
    std::cout << std::endl;

    std::cout << "elementwise_mul_float_naive:" << std::endl;
    elementwise_mul_float_naive(num_elements, iters, grid_dim);
    std::cout << std::endl;

    std::cout << "elementwise_mul_float_optimized:" << std::endl;
    elementwise_mul_float_optimized(num_elements, iters);
    std::cout << std::endl;

    return 0;
}


