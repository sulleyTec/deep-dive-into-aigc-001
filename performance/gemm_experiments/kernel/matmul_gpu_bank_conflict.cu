#include "matmul_gpu_bank_conflict.hpp"
#include "timer.hpp"
#include <stdio.h>
#include <iostream>

static __global__ void gemm_static_naive(float *A, float *B, float *C, int m, int k, int n) {
    __shared__ float A_shared[BLOCKSIZE][BLOCKSIZE];
    __shared__ float B_shared[BLOCKSIZE][BLOCKSIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float tmp = 0.0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // m=n=k
    // A:mxk, B:kxn, C:mxn
    for (int i = 0; i < m / BLOCKSIZE; ++i) {
        A_shared[tx][ty] = A[x + (i*BLOCKSIZE+ty)*m];
        B_shared[tx][ty] = B[(i*BLOCKSIZE+tx) + y*n];
        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; k ++) {
            tmp += A_shared[tx][k] * B_shared[k][ty];
        }

        __syncthreads();
    }

    C[x + y*m] = tmp;
}

/* Question :
   1. will the extra col has effective data, why?
   2. why code commented are not equevalent as current code
 */

static __global__ void gemm_static_padding(float *A, float *B, float *C, int m, int k, int n){
    __shared__ float A_shared[BLOCKSIZE][BLOCKSIZE+1];
    __shared__ float B_shared[BLOCKSIZE][BLOCKSIZE+1];
    // the above should be quevalent to blow, but actually it seems not 
    // (which is equevalent as change major function)
    //__shared__ float A_shared[BLOCKSIZE*(BLOCKSIZE+1)];
    //__shared__ float B_shared[BLOCKSIZE*(BLOCKSIZE+1)];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float tmp = 0.0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // m=n=k
    // A:mxk, B:kxn, C:mxn
    for (int i = 0; i < m / BLOCKSIZE; ++i) {
        //A_shared[ty*BLOCKSIZE+tx] = A[x + (i*BLOCKSIZE+ty)*m];
        //B_shared[tx*BLOCKSIZE+ty] = B[(i*BLOCKSIZE+tx) + y*n];
        A_shared[ty][tx] = A[x + (i*BLOCKSIZE+ty)*m];
        B_shared[tx][ty] = B[(i*BLOCKSIZE+tx) + y*n];
        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; k ++) {
            //tmp += A_shared[k*BLOCKSIZE+tx] * B_shared[k*BLOCKSIZE+ty];
            tmp += A_shared[k][tx] * B_shared[k][ty];
        }

        __syncthreads();
    }

    C[x + y*m] = tmp;
}

static __global__ void gemm_static_change_major(float *A, float *B, float *C, int m, int k, int n){
    __shared__ float A_shared[BLOCKSIZE][BLOCKSIZE];
    __shared__ float B_shared[BLOCKSIZE][BLOCKSIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float tmp = 0.0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // m=n=k
    // A:mxk, B:kxn, C:mxn
    for (int i = 0; i < m / BLOCKSIZE; ++i) {
        A_shared[ty][tx] = A[x + (i*BLOCKSIZE+ty)*m];
        B_shared[tx][ty] = B[(i*BLOCKSIZE+tx) + y*n];
        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; k ++) {
            tmp += A_shared[k][tx] * B_shared[k][ty];
        }

        __syncthreads();
    }

    C[x + y*m] = tmp;
}

static __global__ void gemm_dynamic_bank_conflict(float *A, float *B, float *C, int m){

    extern __shared__ float deviceShared[];
    int stride = BLOCKSIZE * BLOCKSIZE;

    int x = blockIdx.x * BLOCKSIZE + threadIdx.x;
    int y = blockIdx.y * BLOCKSIZE + threadIdx.y;

    float tmp = 0.0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    for (int i = 0; i < m / BLOCKSIZE; i ++) {

        //deviceShared[tx * BLOCKSIZE + ty] = A[x * m + (i * BLOCKSIZE + ty)];
        //deviceShared[stride + (tx * BLOCKSIZE + ty)]  = B[(i * BLOCKSIZE+ tx) * m + y];

        deviceShared[tx+ty*BLOCKSIZE] = A[x * m + (i * BLOCKSIZE + ty)];
        deviceShared[stride + (tx * BLOCKSIZE + ty)]  = B[(i * BLOCKSIZE+ tx) * m + y];

        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; k ++) {
            tmp += deviceShared[tx * BLOCKSIZE + k] * deviceShared[stride + (k * BLOCKSIZE+ ty)];
        }
        __syncthreads();
    }

    /* 列优先 */
    C[x * m + y] = tmp;
}

void gemm_bank_conflict(float* A, float* B, float* C, 
                        int m, int k, int n, 
                        cudaStream_t &stream, ConflictType ct){

    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 dimGrid(m / BLOCKSIZE, m / BLOCKSIZE);

    long int sMemSize = BLOCKSIZE*BLOCKSIZE*sizeof(float)*2;

    //Timer timer;
    //timer.start(stream);

    switch (ct) {
        case ConflictType::naive:
            gemm_static_naive<<<dimGrid,dimBlock,0,stream>>>(A, B, C, m, k, n);
            break;
        case ConflictType::padding:
            gemm_static_padding<<<dimGrid,dimBlock,0,stream>>>(A, B, C, m, k, n);
            break;
        case ConflictType::change_major:
            gemm_static_change_major<<<dimGrid,dimBlock,0,stream>>>(A, B, C, m, k, n);
            break;
        case ConflictType::dynamic:
            gemm_dynamic_bank_conflict<<<dimGrid, dimBlock, sMemSize, stream>>> (A, B, C, m);
            break;
        default:
            std::cout << "invalid type" << std::endl;
    }

    //timer.stop("calc time");
}

