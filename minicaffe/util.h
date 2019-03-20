/***
 * @file util.h
 * @author Quan Fan
 * @brief Utilities functions.
 * @date 01/Mar/2019
 */

#ifndef _UTIL_H_
#define _UTIL_H_

#include <cmath>
#include <iostream>
#include "blob.h"
void alloc_and_strcpy(char**dst, const char* src);

int print_matrix(float *A, int batch_size, int x, int y, int z);

int random_matrix(float *A, int eleNum, float scalar=1);

typedef struct coordinate
{
	int row;
	int col;
	int z;
}coord, *coord_ptr;

int print_coord_matrix(coord_ptr A, int batch_size, int x, int y, int z);


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


namespace helper{
    inline bool float_eq(float a, float b){
        if (std::abs(a-b)<0.0001) return 1;
        return 0;
    }

    static void print_blob(Blob &blob){
        using namespace std;
        cout<<"print blob "<<blob.name<<endl;
        for(int batch=0;batch<blob.batchSize;batch++){
            cout<<"batch "<<batch<<endl;
            for(int channel=0;channel<blob.z;channel++){
                cout<<"channel "<<channel<<endl;
                for(int y=0;y<blob.y;y++){
                    for(int x=0;x<blob.x;x++){
                        cout<<blob(batch,x,y,channel)<<" ";
                    }
                    cout<<endl;
                }

            }
        }
        cout<<endl;
    }
}

#endif