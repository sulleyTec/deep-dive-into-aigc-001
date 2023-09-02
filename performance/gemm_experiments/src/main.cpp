#include <iostream>
#include <random>
#include <cublas_v2.h>

#include "memory.hpp"
#include "matmul_gpu_bank_conflict.hpp"

void randomize_matrix(float* mat, int N){
    srand(time(NULL)); int i;
    for (i = 0; i < N; i++) {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        //tmp = i;
        mat[i] = tmp;
    }
}

bool verify_matrix(float *mat1, float *mat2, int n){
    double diff = 0.0;
    int i;
    for (i = 0; mat1 + i && mat2 + i && i < n; i++){
        diff = fabs( (double)mat1[i] - (double)mat2[i] );
        if (diff > 1e-2) {
            printf("error. %5.2f, %5.2f,%d\n", mat1[i],mat2[i],i);
            return false;
        }
    }
    return true;
}

int main() {
    int m,n,k;
    m=n=k=1<<12;
    float alpha = 1.0, beta = 0.;//two arbitary input parameters

    int size = m*n;
    Memory<float> A, B, C, C_ref;
    int loop_num = 1;

    // malloc memory both in gpu and cpu
    float *A_d = A.gpu(size);
    float *A_h = A.cpu(size);

    float *B_d = B.gpu(size);
    float *B_h = B.cpu(size);

    float *C_d = C.gpu(size);
    float *C_h = C.cpu(size);

    float *C_ref_d = C_ref.gpu(size);
    float *C_ref_h = C_ref.cpu(size);

    // Initialize input matrix with data
    std::cout << "init" << std::endl;
    randomize_matrix(A_h, size);
    randomize_matrix(B_h, size);

    std::cout << "init done" << std::endl;

    cudaMemcpy(A_d,
               A_h,
               size * sizeof(float), 
               cudaMemcpyHostToDevice);

    cudaMemcpy(B_d,
               B_h,
               size * sizeof(float), 
               cudaMemcpyHostToDevice);

    std::cout << "start calc" << std::endl;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasHandle_t err; cublasCreate(&err);

    std::cout << std::endl;
    std::cout << "warm up" << std::endl;
    for(int i = 0; i<2; ++i) {

        /* A:mxk, B:kxn, C:mxn */
        cublasSgemm(err, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 
                    &alpha, A_d, m, B_d, k, &beta, C_ref_d, m);

        gemm_bank_conflict(A_d, B_d, C_d, m, k, n, stream, 
                           ConflictType::naive);

        cudaMemcpy(C_h, C_d, 
                   size * sizeof(float), 
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(C_ref_h, C_ref_d, 
                   size * sizeof(float), 
                   cudaMemcpyDeviceToHost);


        if (!verify_matrix(C_ref_h,C_h,m*n)) {
            printf("Failed to pass the correctness verification \
                   against NVIDIA cuBLAS. Exited.\n");
            exit(-3);
        }
    }

    std::cout << std::endl;
    std::cout << "naive" << std::endl;
    for(int i = 0; i<loop_num; ++i) {

        /* A:mxk, B:kxn, C:mxn */
        cublasSgemm(err, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 
                    &alpha, A_d, m, B_d, k, &beta, C_ref_d, m);

        gemm_bank_conflict(A_d, B_d, C_d, m, k, n, stream, 
                           ConflictType::naive);

        cudaMemcpy(C_h, C_d, 
                   size * sizeof(float), 
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(C_ref_h, C_ref_d, 
                   size * sizeof(float), 
                   cudaMemcpyDeviceToHost);


        if (!verify_matrix(C_ref_h,C_h,m*n)) {
            printf("Failed to pass the correctness verification \
                   against NVIDIA cuBLAS. Exited.\n");
            exit(-3);
        }
    }

    std::cout << std::endl;
    std::cout << "padding" << std::endl;
    for(int i = 0; i<loop_num; ++i) {

        /* A:mxk, B:kxn, C:mxn */
        cublasSgemm(err, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 
                    &alpha, A_d, m, B_d, k, &beta, C_ref_d, m);

        gemm_bank_conflict(A_d, B_d, C_d, m, k, n, stream, 
                           ConflictType::padding);

        cudaMemcpy(C_h, C_d, 
                   size * sizeof(float), 
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(C_ref_h, C_ref_d, 
                   size * sizeof(float), 
                   cudaMemcpyDeviceToHost);


        if (!verify_matrix(C_ref_h,C_h,m*n)) {
            printf("Failed to pass the correctness verification \
                   against NVIDIA cuBLAS. Exited.\n");
            exit(-3);
        }
    }

    std::cout << std::endl;
    std::cout << "change major" << std::endl;
    for(int i = 0; i<loop_num; ++i) {

        /* A:mxk, B:kxn, C:mxn */
        cublasSgemm(err, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 
                    &alpha, A_d, m, B_d, k, &beta, C_ref_d, m);

        gemm_bank_conflict(A_d, B_d, C_d, m, k, n, stream, 
                           ConflictType::change_major);

        cudaMemcpy(C_h, C_d, 
                   size * sizeof(float), 
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(C_ref_h, C_ref_d, 
                   size * sizeof(float), 
                   cudaMemcpyDeviceToHost);


        if (!verify_matrix(C_ref_h,C_h,m*n)) {
            printf("Failed to pass the correctness verification \
                   against NVIDIA cuBLAS. Exited.\n");
            exit(-3);
        }
    }

    std::cout << "done" << std::endl;

    return 0;
}

