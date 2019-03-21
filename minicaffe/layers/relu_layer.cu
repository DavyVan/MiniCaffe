#include "relu_layer.h"
#include <assert.h>

__global__ void relu_infer(float *out,float *in, unsigned size){
    unsigned idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<size){
        out[idx]=max(0.0,in[idx]);
    }
}
__global__ void relu_bp(float *out,float *left_in,float *right_in, unsigned size){
    unsigned idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<size){
        out[idx]=left_in[idx]>0?right_in[idx]:0;
    }
}


void ReluLayer::infer_gpu(vector<Blob *> lefts, vector<Blob *> rights) {
    rights[0]->reset();
    unsigned threads=256;
    int num_ele=lefts[0]->get_ele_num();

    float *in_h,*out_h,*in_d,*out_d;
    cudaError_t cuda_ret;

    in_h=lefts[0]->_data;
    out_h=new float[num_ele];//(float*)malloc(num_ele*sizeof(float));
    cuda_ret = cudaMalloc((void**)&in_d, num_ele * sizeof(float));
    assert(cuda_ret == cudaSuccess);
    cuda_ret = cudaMalloc((void**)&out_d, num_ele * sizeof(float));
    if(cuda_ret == cudaSuccess);
    cudaDeviceSynchronize();

    cuda_ret = cudaMemcpy(in_d, in_h, num_ele * sizeof(float),
                          cudaMemcpyHostToDevice);
    assert(cuda_ret == cudaSuccess);
    cuda_ret = cudaMemset(out_d, 0, num_ele * sizeof(float));
    assert(cuda_ret == cudaSuccess);
    cudaDeviceSynchronize();

    dim3 dim_grid, dim_block;
    dim_block.x=threads;
    dim_block.y=dim_block.z=1;
    dim_grid.x=((num_ele-1)/threads+1);
    dim_grid.y=1;
    dim_grid.z=1;
    relu_infer<<<dim_grid,dim_block>>>(out_d,in_d,num_ele);
    cuda_ret = cudaDeviceSynchronize();
    assert(cuda_ret == cudaSuccess);

    cuda_ret = cudaMemcpy(out_h, out_d, num_ele * sizeof(float),
                          cudaMemcpyDeviceToHost);
    assert(cuda_ret == cudaSuccess);
    cudaDeviceSynchronize();

    delete[] rights[0]->_data;
    rights[0]->_data=out_h;
    cudaFree(in_d);
    cudaFree(out_d);
}

void ReluLayer::bp_gpu(vector<Blob *> lefts, vector<Blob *> rights) {

    unsigned threads=256;
    int num_ele=lefts[0]->get_ele_num();

    float *left_h,*right_h,*out_h;
    float *left_d,*right_d,*out_d;
    cudaError_t cuda_ret;

    right_h=rights[0]->_data;
    left_h=lefts[0]->_data;

    out_h=new float[num_ele]; //(float*)malloc(num_ele*sizeof(float));

    cuda_ret = cudaMalloc((void**)&left_d, num_ele * sizeof(float));
    assert(cuda_ret == cudaSuccess);
    cuda_ret = cudaMalloc((void**)&right_d, num_ele * sizeof(float));
    assert(cuda_ret == cudaSuccess);
    cuda_ret = cudaMemcpy(left_d, left_h, num_ele * sizeof(float),cudaMemcpyHostToDevice);
    assert(cuda_ret == cudaSuccess);
    cuda_ret = cudaMemcpy(right_d, right_h, num_ele * sizeof(float),cudaMemcpyHostToDevice);
    assert(cuda_ret == cudaSuccess);

    cuda_ret = cudaMalloc((void**)&out_d, num_ele * sizeof(float));
    if(cuda_ret == cudaSuccess);
    cudaDeviceSynchronize();

    cuda_ret = cudaMemset(out_d, 0, num_ele * sizeof(float));
    assert(cuda_ret == cudaSuccess);
    cudaDeviceSynchronize();

    dim3 dim_grid, dim_block;
    dim_block.x=threads;
    dim_block.y=dim_block.z=1;
    dim_grid.x=((num_ele-1)/threads+1);
    dim_grid.y=1;
    dim_grid.z=1;
    relu_bp<<<dim_grid,dim_block>>>(out_d,left_d,right_d,num_ele);
    cuda_ret = cudaDeviceSynchronize();
    assert(cuda_ret == cudaSuccess);

    cuda_ret = cudaMemcpy(out_h, out_d, num_ele * sizeof(float),
                          cudaMemcpyDeviceToHost);
    assert(cuda_ret == cudaSuccess);
    cudaDeviceSynchronize();

    delete[] lefts[0]->_data;
    lefts[0]->_data=out_h;
    cudaFree(left_d);
    cudaFree(right_d);
    cudaFree(out_d);
}
