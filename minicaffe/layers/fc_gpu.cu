#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "fc_layer.h"
#include "../seqnet.h"
#include "../util.h"
#include "../errors.h"


__global__ void mysgemm(int m, int n, int k, const float alpha, const float *A, const float *B, const float beta, float* C) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        float tmp = beta * C[row*n+col];
        for(int i = 0; i < k; i++)
        {
            tmp += alpha * A[row*k+i]*B[i*n+col];
        }
        C[row * n + col] = tmp;
    }
}

void basicSgemm(int m, int n, int k, float alpha, const float *A, const float *B, float beta, float *C)
{
    const unsigned int BLOCK_SIZE = 512;

    dim3 dimGrid( (n-1)/BLOCK_SIZE + 1, (m-1)/BLOCK_SIZE+1, 1 );
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    mysgemm<<<dimGrid, dimBlock>>>(m, n, k, alpha, A, B, beta, C);
}

__global__ void VecAdd(int n, const float *A, const float *B, float* C) {

    int localIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (localIdx < n)
    {
        C[localIdx] = A[localIdx] + B[localIdx];
    } 
}


void basicVecAdd( float *A,  float *B, float *C, int n)
{

    const unsigned int BLOCK_SIZE = 512;

    dim3 DimGrid((n - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    VecAdd<<<DimGrid,DimBlock>>>(n, A, B, C);
}

void FCLayer::infer_gpu(std::vector<Blob*> left_blobs, std::vector<Blob*> right_blobs)
{
    // treat _data as a 2D matrix
    // right = left * weight
    float *in_h, *out_h, *in_d, *out_d, *weight_h, *weight_d, *bias_d;
    int num_in, num_out, num_weight;
    cudaError_t cuda_ret;

    num_in = M_ * K_;
    num_out = M_ * N_;
    num_weight = K_ * N_;

    in_h = left_blobs[0]->_data;
    out_h = (float*)malloc(num_out * sizeof(float));
    weight_h = weight;

    cudaMalloc((void**)&in_d, num_in * sizeof(float));
    cudaMalloc((void**)&out_d, num_out * sizeof(float));
    cudaMalloc((void**)&weight_d, num_weight * sizeof(float));
    cudaMalloc((void**)&bias_d, N_ * sizeof(float));

    cudaDeviceSynchronize();

    cudaMemcpy(in_d, in_h, num_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out_h, num_out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_d, weight_h, num_weight * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_d, bias, N_ * sizeof(float), cudaMemcpyHostToDevice);

    basicSgemm(M_, N_, K_, 1, in_d, weight_d, 0, out_d);

    cudaDeviceSynchronize();

    // bias
    if (bias_term)
    {
        for (int b = 0; b < M_; b++)
        {
            basicVecAdd(out_d + b * N_, bias_d, out_d + b * N_, N_);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(out_h, out_d, num_out * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    if (right_blobs[0]->_data) free(right_blobs[0]->_data);
    right_blobs[0]->_data = out_h;

    cudaFree(in_d);
    cudaFree(out_d);
    cudaFree(weight_d);
    cudaFree(bias_d);
}

void FCLayer::bp_gpu(std::vector<Blob*> lefts, std::vector<Blob*> rights)
{
    Blob* left = lefts[0];
    Blob* right = rights[0];

    // weight
    // we first transpose left from MxK --> KxM
    float* leftT = new float[K_ * M_];
    for (int row = 0; row < M_; row++)
    {
        for (int col = 0; col < K_; col++)
        {
            leftT[col * M_ + row] = left->_data[row * K_ + col];
        }
    }

    float *left_d, *left_h, *left_t_h, *left_t_d, *bias_h, *bias_d, *right_h, *right_d, *weight_h, *weight_d;
    int num_left, num_right, num_weight, num_bias;
    cudaError_t cuda_ret;

    num_left = M_ * K_;
    num_right = M_ * N_;
    num_weight = K_ * N_;
    num_bias = N_;

    left_h = left->_data;
    left_t_h = leftT;
    bias_h = bias;
    weight_h = weight;
    right_h = right->_data;

    cudaMalloc((void**)&left_d, num_left * sizeof(float));
    cudaMalloc((void**)&right_d, num_right * sizeof(float));
    cudaMalloc((void**)&weight_d, num_weight * sizeof(float));
    cudaMalloc((void**)&bias_d, num_bias * sizeof(float));
    cudaMalloc((void**)&left_t_d, num_left * sizeof(float));
    
    cudaMemcpy(left_d, left_h, num_left * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(right_d, right_h, num_right * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_d, weight_h, num_weight * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_d, bias_h, num_bias * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(left_t_d, left_t_h, num_left * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    basicSgemm(K_, N_, M_, 1, left_t_d, right_d, 1, weight_d);

    // bias
    if (bias_term)
    {
        for (int b = 0; b < M_; b++)
        {
            basicVecAdd(right_d + b * N_, bias_d, bias_d, N_);
            cudaDeviceSynchronize();
        }
    }

    cuda_ret = cudaDeviceSynchronize();
    {
        if (cuda_ret != cudaSuccess)
            printf("errors occured!\n");
    }
    basicSgemm(M_, K_, N_, 1, right_d, weight_d, 0, left_d);
    cudaDeviceSynchronize();

    cuda_ret = cudaMemcpy(left_h, left_d, num_left * sizeof(float),
                          cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
    {
        print_err_str(ZERO_DIM);
        exit(ZERO_DIM);
    }

    cudaDeviceSynchronize();

    if (lefts[0]->_data) free(lefts[0]->_data);
    lefts[0]->_data = left_h;

    cudaFree(left_d);
    cudaFree(left_t_d);
    cudaFree(right_d);
    cudaFree(weight_d);
    cudaFree(bias_d);

    delete[] leftT;
}
