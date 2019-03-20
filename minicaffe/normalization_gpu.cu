#include "normalization.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void normalization_infer_basic_kernel(
	float *input,
	float *output,
	int in_x,
	int in_y,
	int in_z,
	int batch_size,
	int nn,
	int kk,
	float alpha,
	float beta
	)
{
	float *input_curr, *output_curr;
	int t, curr_idx;

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int z   = blockDim.z * blockIdx.z + threadIdx.z;

    if (row >= in_y || col >= in_x || z >= in_z) { return; }

    float curr_sum, inner_value;
    int start, end, idx;

    for (curr_idx = 0; curr_idx < batch_size; curr_idx++)
    {
    	input_curr = input + curr_idx * in_x * in_y * in_z;
    	output_curr = output + curr_idx * in_x * in_y * in_z;

		start = ( (z - nn / 2) < 0 ) ? 0 : (z - nn / 2);
		end = ( (in_z - 1) < (z + nn / 2) ) ? (in_z - 1) : (z + nn / 2);

		curr_sum = 0.0;
		for (t = start; t <= end; t++) 
		{
			curr_sum += input_curr[(t * in_y + row) * in_x + col] * input_curr[(t * in_y + row) * in_x + col];
		}

		idx = (z * in_y + row) * in_x + col;
		inner_value = kk + alpha * curr_sum;
		output_curr[idx] = input_curr[idx] / pow(inner_value, beta);
    }
}

__global__ void normalization_bp_basic_kernel(
	float *input,
	float *output,
	int in_x,
	int in_y,
	int in_z,
	int batch_size,
	int nn,
	int kk,
	float alpha,
	float beta,
	float *old
	)
{
	float *input_curr, *output_curr, *old_curr;
	int t, curr_idx;

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int z   = blockDim.z * blockIdx.z + threadIdx.z;

    if (row >= in_y || col >= in_x || z >= in_z) { return; }

    float curr_sum_cf, curr_sum_cfdf, inner_value;
    int start, end, idx;

    for (curr_idx = 0; curr_idx < batch_size; curr_idx++)
    {
    	input_curr = input + curr_idx * in_x * in_y * in_z;
    	old_curr = old + curr_idx * in_x * in_y * in_z;
    	output_curr = output + curr_idx * in_x * in_y * in_z;

		start = ( (z - nn / 2) < 0 ) ? 0 : (z - nn / 2);
		end = ( (in_z - 1) < (z + nn / 2) ) ? (in_z - 1) : (z + nn / 2);

		curr_sum_cf = 0.0;
		curr_sum_cfdf = 0.0;
		for (t = start; t <= end; t++) 
		{
			curr_sum_cf += old_curr[(t * in_y + row) * in_x + col] * old_curr[(t * in_y + row) * in_x + col];
			curr_sum_cfdf += input_curr[(t * in_y + row) * in_x + col] * old_curr[(t * in_y + row) * in_x + col];
		}

		idx = (z * in_y + row) * in_x + col;
		inner_value = kk + alpha * curr_sum_cf;
		output_curr[idx] = input_curr[idx] * pow(inner_value, -1.0 * beta) - 2 * beta * alpha * pow(inner_value, -1.0 * beta - 1)
									  * curr_sum_cfdf * old_curr[idx];
    }
}


void NormalizationLayer::infer_gpu(vector<Blob *> left_blobs, vector<Blob *> right_blobs)
{
	int numInputs = left_blobs.size();

	if (numInputs != 1) 
	{
		print_err_str(LEFT_NOT_MATCH);
		exit(LEFT_NOT_MATCH);
	}

    right_blobs[0]->reset();
    unsigned threads = 4;
    int num_ele = left_blobs[0]->get_ele_num();
    

	float *in_h, *out_h, *in_d, *out_d;
	int in_x, in_y, in_z;
	cudaError_t cuda_ret;

	in_x = left_blobs[0]->x;
	in_y = left_blobs[0]->y;
	in_z = left_blobs[0]->z;

	in_h = left_blobs[0]->_data;
	out_h = (float*)malloc(num_ele*sizeof(float));

    cuda_ret = cudaMalloc((void**)&in_d, num_ele * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cuda_ret = cudaMalloc((void**)&out_d, num_ele * sizeof(float));
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

    cuda_ret = cudaMemset(out_d, 0, num_ele * sizeof(float));
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

    normalization_infer_basic_kernel<<<dim_grid,dim_block>>>(in_d, out_d, in_x, in_y, in_z, batch_size, nn, kk, alpha, beta);

    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cuda_ret = cudaMemcpy(out_h, out_d, num_ele * sizeof(float),
                          cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cudaDeviceSynchronize();

    if (right_blobs[0]->_data) free(right_blobs[0]->_data);
    right_blobs[0]->_data = out_h;

    cudaFree(in_d);
    cudaFree(out_d);

}

void NormalizationLayer::bp_gpu(vector<Blob *> left_blobs, vector<Blob *> right_blobs)
{
	int numInputs = right_blobs.size();

	if (numInputs != 1) 
	{
		print_err_str(LEFT_NOT_MATCH);
		exit(LEFT_NOT_MATCH);
	}

    Blob back_up_left = *(left_blobs[0]);
    memcpy(back_up_left._data, left_blobs[0]->_data, left_blobs[0]->get_ele_num() * sizeof(float));

    unsigned threads = 4;
    int num_ele = right_blobs[0]->get_ele_num();
    

	float *in_h, *out_h, *in_d, *out_d, *old_h, *old_d;
	int in_x, in_y, in_z;
	cudaError_t cuda_ret;

	in_x = right_blobs[0]->x;
	in_y = right_blobs[0]->y;
	in_z = right_blobs[0]->z;

	in_h = right_blobs[0]->_data;
	old_h = back_up_left._data;
	out_h = (float*)malloc(num_ele * sizeof(float));

    cuda_ret = cudaMalloc((void**)&in_d, num_ele * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cuda_ret = cudaMalloc((void**)&old_d, num_ele * sizeof(float));
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cuda_ret = cudaMalloc((void**)&out_d, num_ele * sizeof(float));
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

    cuda_ret = cudaMemcpy(old_d, old_h, num_ele * sizeof(float),
                          cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cuda_ret = cudaMemset(out_d, 0, num_ele * sizeof(float));
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

    normalization_bp_basic_kernel<<<dim_grid,dim_block>>>(in_d, out_d, in_x, in_y, in_z, batch_size, nn, kk, alpha, beta, old_d);

    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cuda_ret = cudaMemcpy(out_h, out_d, num_ele * sizeof(float),
                          cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess)
    {
		print_err_str(ZERO_DIM);
		exit(ZERO_DIM);
    }

    cudaDeviceSynchronize();

    if (left_blobs[0]->_data) free(left_blobs[0]->_data);
    left_blobs[0]->_data = out_h;

    cudaFree(in_d);
    cudaFree(out_d);

}
