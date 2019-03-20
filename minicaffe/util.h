/***
 * @file util.h
 * @author Quan Fan
 * @brief Utilities functions.
 * @date 01/Mar/2019
 */
#include <cmath>
void alloc_and_strcpy(char**dst, const char* src);

int print_matrix(float *A, int batch_size, int x, int y, int z);

int random_matrix(float *A, int eleNum);

typedef struct coordinate
{
	int row;
	int col;
	int z;
}coord, *coord_ptr;

int print_coord_matrix(coord_ptr A, int batch_size, int x, int y, int z);

namespace helper{
    inline bool float_eq(float a, float b){
        if (std::abs(a-b)<0.0001) return 1;
        return 0;
    }
}

/***
 * @brief C[MxN] = alpha * A[MxK] * B[KxN] + beta * C[MxN]
 *        A or B cannot be C
 * 
 */
void simple_gemm(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);

/***
 * @brief C = A + B, C can be A or B
 * 
 */
void vector_add(int L, float* A, float* B, float* C);