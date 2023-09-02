
#define BLOCKSIZE 32
//#define BLOCKSIZE 16
//#define BLOCKSIZE 8

enum class ConflictType{
    naive,
    padding,
    change_major,
    dynamic,
};

void gemm_bank_conflict(float* A, float* B, float* C, 
                        int m, int k, int n, 
                        cudaStream_t &stream, ConflictType ct);

