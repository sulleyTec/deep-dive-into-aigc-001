#include <iostream>
#include <stdio.h>
#include <cuda_fp16.h>
#include "elementwise.hpp"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <num_elements> <grid_dim> <function_name>" << std::endl;
        return 1;
    }

    //const int64_t num_elements = 163840*1000; 
    int iters = 10;

    const int64_t num_elements = std::stoll(argv[1]);
    int grid_dim = std::atoi(argv[2]);
    std::string functionName = argv[3];

    printf("performance test: num_elements=%ld, grid_dim=%d\n", 
           num_elements, grid_dim);


    if (functionName == "elementwise_add_half_naive") {
        std::cout << "elementwise_add_half_naive:" << std::endl;
        elementwise_add_half_naive(num_elements, iters, grid_dim);
    } else if (functionName == "elementwise_add_float_naive") {
        std::cout << "elementwise_add_float_naive:" << std::endl;
        elementwise_add_float_naive(num_elements, iters, grid_dim);
    } else if (functionName == "elementwise_add_half_optimized") {
        std::cout << "elementwise_add_half_optimized:" << std::endl;
        elementwise_add_half_optimized(num_elements, iters);
    } else if (functionName == "elementwise_add_float_optimized") {
        std::cout << "elementwise_add_float_optimized:" << std::endl;
        elementwise_add_float_optimized(num_elements, iters);
    } else if (functionName == "elementwise_mul_half_naive") {
        std::cout << "elementwise_mul_half_naive:" << std::endl;
        elementwise_mul_half_naive(num_elements, iters, grid_dim);
    } else if (functionName == "elementwise_mul_float_naive") {
        std::cout << "elementwise_mul_float_naive:" << std::endl;
        elementwise_mul_float_naive(num_elements, iters, grid_dim);
    } else if (functionName == "elementwise_mul_half_optimized") {
        std::cout << "elementwise_mul_half_optimized:" << std::endl;
        elementwise_mul_half_optimized(num_elements, iters);
    } else if (functionName == "elementwise_mul_float_optimized") {
        std::cout << "elementwise_mul_float_optimized:" << std::endl;
        elementwise_mul_float_optimized(num_elements, iters);
    } else {
        std::cerr << "Invalid function name: " << functionName << std::endl;
        return 1;
    }

    std::cout << std::endl;
    return 0;
}


