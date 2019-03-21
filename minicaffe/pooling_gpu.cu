#include "pooling.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void pooling_infer_basic_kernel(
	float *input,
	float *output,
	int in_x,
	int in_y,
	int in_z,
	int batch_size,
	int mask_x,
	int mask_y,
	int stride,
	int out_x,
	int out_y,
	int out_z,
	coord_ptr tmp_coord
	)
{
	float *input_curr, *output_curr;
	coord_ptr tmp_coord_curr; 
	int i, j, curr_idx;

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int z   = blockDim.z * blockIdx.z + threadIdx.z;

//	printf("col = %d, row = %d, z = %d", col, row, z);

    if (row >= out_y || col >= out_x || z >= out_z) { return; }

    float curr_max = -1;
    int curr_idx_x, curr_idx_y;

    for (curr_idx = 0; curr_idx < batch_size; curr_idx++)
    {
    	input_curr = input + curr_idx * in_x * in_y * in_z;
    	output_curr = output + curr_idx * out_x * out_y * out_z;
    	tmp_coord_curr = tmp_coord + curr_idx * out_x * out_y * out_z;
	    for (i = row * stride; i < row * stride + mask_y && i < in_y; i++)
	    {
	    	for (j = col * stride; j < col * stride + mask_x && j < in_x; j++)
	    	{
	    		if (curr_max < input_curr[(z * in_y + i) * in_x + j])
	    		{
//	    			printf("col = %d, row = %d, z = %d, i = %d, j = %d, input_curr = %f\n", col, row, z, i, j, input_curr[(z * m + i) * n + j]);
	    			curr_idx_y = i;
	    			curr_idx_x = j;
	    			curr_max = input_curr[(z * in_y + i) * in_x + j];
	    		}
	    	}
	    }

	    output_curr[(z * out_y + row) * out_x + col] = curr_max;
	    tmp_coord_curr[(z * out_y + row) * out_x + col].col = curr_idx_x;
	    tmp_coord_curr[(z * out_y + row) * out_x + col].row = curr_idx_y;
	    tmp_coord_curr[(z * out_y + row) * out_x + col].z = z;

    }
}


__global__ void pooling_bp_basic_kernel(
	float *input,
	float *output,
	int in_x,
	int in_y,
	int in_z,
	int batch_size,
	int mask_x,
	int mask_y,
	int stride,
	int out_x,
	int out_y,
	int out_z,
	coord_ptr tmp_coord
	)
{
	float *input_curr, *output_curr;
	coord_ptr tmp_coord_curr; 
	int curr_idx;

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int z   = blockDim.z * blockIdx.z + threadIdx.z;

//	printf("col = %d, row = %d, z = %d", col, row, z);

    if (col >= in_x || row >= in_y || z >= in_z) { return; }

    int curr_idx_x, curr_idx_y;

    for (curr_idx = 0; curr_idx < batch_size; curr_idx++)
    {
    	input_curr = input + curr_idx * in_x * in_y * in_z;
    	output_curr = output + curr_idx * out_x * out_y * out_z;
    	tmp_coord_curr = tmp_coord + curr_idx * in_x * in_y * in_z;

    	curr_idx_x = tmp_coord_curr[(z * in_y + row) * in_x + col].col;
    	curr_idx_y = tmp_coord_curr[(z * in_y + row) * in_x + col].row;

	    output_curr[(z * out_y + curr_idx_y) * out_x + curr_idx_x] = input_curr[(z * in_y + row) * in_x + col];

    }
}


void PoolingLayer::infer_gpu(vector<Blob *> left_blobs, vector<Blob *> right_blobs)
{
  infer(left_blobs, right_blobs);
  return;
	int numInputs = left_blobs.size();

	if (numInputs != 1) 
	{
		print_err_str(LEFT_NOT_MATCH);
		exit(LEFT_NOT_MATCH);
	}

    right_blobs[0]->reset();
    unsigned threads = 256;
    int num_ele = left_blobs[0]->get_ele_num();
    

	float *in_h, *out_h, *in_d, *out_d;
	coord_ptr tmp_space_d;
	int in_x, in_y, in_z;
	cudaError_t cuda_ret;

	in_x = left_blobs[0]->x;
	in_y = left_blobs[0]->y;
	in_z = left_blobs[0]->z;

    int out_x = right_blobs[0]->x;
    int out_y = right_blobs[0]->y;
    int out_z = right_blobs[0]->z;
    int out_ele = right_blobs[0]->get_ele_num();

	in_h = left_blobs[0]->_data;
	out_h = new float[out_ele]; //(float*)malloc(out_ele * sizeof(float));

    cuda_ret = cudaMalloc((void**)&in_d, num_ele * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cuda_ret = cudaMalloc((void**)&out_d, out_ele * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cudaDeviceSynchronize();

    cuda_ret = cudaMemcpy(in_d, in_h, num_ele * sizeof(float),
                          cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cuda_ret = cudaMemset(out_d, 0, out_ele * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cudaDeviceSynchronize();

    dim3 dim_grid, dim_block;
    dim_block.x = threads;
    dim_block.y = threads;
    dim_block.z = threads;

    dim_grid.x = ( (out_x - 1) / threads+1 );
    dim_grid.y = ( (out_y - 1) / threads+1 );
    dim_grid.z = ( (out_z - 1) / threads+1 );

	if (tmp_space) free(tmp_space);

	tmp_space = (coordinate *)malloc(sizeof(coordinate) * out_ele);

    cuda_ret = cudaMalloc((void**)&tmp_space_d, sizeof(coordinate) * out_ele);
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    int batch_size = left_blobs[0]->batchSize;
/*
	err = print_matrix(in_h, batch_size, in_x, in_y, in_z);
	if (err != 0)
	{
		printf("%d\n", err);
	}
*/
    pooling_infer_basic_kernel<<<dim_grid,dim_block>>>(in_d, out_d, in_x, in_y, in_z, batch_size, mask_x, mask_y, stride, out_x, out_y, out_z, tmp_space_d);

    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cuda_ret = cudaMemcpy(out_h, out_d, out_ele * sizeof(float),
                          cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cuda_ret = cudaMemcpy(tmp_space, tmp_space_d, out_ele * sizeof(coordinate),
                          cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cudaDeviceSynchronize();

    if (right_blobs[0]->_data) delete[] right_blobs[0]->_data;
    right_blobs[0]->_data=out_h;

    cudaFree(in_d);
    cudaFree(out_d);
    cudaFree(tmp_space_d);

}


void PoolingLayer::bp_gpu(vector<Blob *> left_blobs, vector<Blob *> right_blobs)
{
	int numInputs = right_blobs.size();

	if (numInputs != 1) 
	{
		print_err_str(LEFT_NOT_MATCH);
		exit(LEFT_NOT_MATCH);
	}

    left_blobs[0]->reset();
    unsigned threads = 256;
    int num_ele = right_blobs[0]->get_ele_num();

	float *in_h, *out_h, *in_d, *out_d;
	coord_ptr tmp_space_d, tmp_space_h;
	int in_x, in_y, in_z, out_x, out_y, out_z;
	cudaError_t cuda_ret;

	out_x = left_blobs[0]->x;
	out_y = left_blobs[0]->y;
	out_z = left_blobs[0]->z;

    in_x = right_blobs[0]->x;
    in_y = right_blobs[0]->y;
    in_z = right_blobs[0]->z;

    int out_ele = left_blobs[0]->get_ele_num();

	in_h = right_blobs[0]->_data;
	tmp_space_h = tmp_space;
	out_h = new float[out_ele]; //(float*)malloc(out_ele * sizeof(float));

    cuda_ret = cudaMalloc((void**)&in_d, num_ele * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cuda_ret = cudaMalloc((void**)&out_d, out_ele * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cudaDeviceSynchronize();

    cuda_ret = cudaMemcpy(in_d, in_h, num_ele * sizeof(float),
                          cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cuda_ret = cudaMalloc((void**)&tmp_space_d, num_ele * sizeof(coordinate));
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cuda_ret = cudaMemcpy(tmp_space_d, tmp_space_h, num_ele * sizeof(coordinate),
                          cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cuda_ret = cudaMemset(out_d, 0, out_ele * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cudaDeviceSynchronize();

    dim3 dim_grid, dim_block;
    dim_block.x = threads;
    dim_block.y = threads;
    dim_block.z = threads;

    dim_grid.x = ( (in_x - 1) / threads+1 );
    dim_grid.y = ( (in_y - 1) / threads+1 );
    dim_grid.z = ( (in_z - 1) / threads+1 );


    int batch_size = left_blobs[0]->batchSize;
/*
	err = print_matrix(in_h, batch_size, in_x, in_y, in_z);
	if (err != 0)
	{
		printf("%d\n", err);
	}
*/
    pooling_bp_basic_kernel<<<dim_grid,dim_block>>>(in_d, out_d, in_x, in_y, in_z, batch_size, mask_x, mask_y, stride, out_x, out_y, out_z, tmp_space_d);

    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cuda_ret = cudaMemcpy(out_h, out_d, out_ele * sizeof(float),
                          cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cudaDeviceSynchronize();

    if (left_blobs[0]->_data) delete[] left_blobs[0]->_data;
    left_blobs[0]->_data = out_h;

    if (tmp_space) free(tmp_space);

    cudaFree(in_d);
    cudaFree(out_d);
    cudaFree(tmp_space_d);

}



