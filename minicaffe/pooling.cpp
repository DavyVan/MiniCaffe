#include "pooling.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

PoolingLayer::PoolingLayer():Layer("undefined")
{
	mask_x = 5;
	mask_y = 5;
	stride = 2;
	tmp_space = NULL;
}

PoolingLayer::PoolingLayer(char *name):Layer(name)
{
	mask_x = 5;
	mask_y = 5;
	stride = 2;
	tmp_space = NULL;
}

PoolingLayer::PoolingLayer(char *name, int mask_x_, int mask_y_, int stride_):Layer(name)
{
	mask_x = mask_x_;
	mask_y = mask_y_;
	stride = stride_;
	tmp_space = NULL;
}

PoolingLayer::~PoolingLayer()
{
	;
}

int PoolingLayer::init() {return 0;}

void PoolingLayer::infer(vector<Blob*> left_blobs, vector<Blob*> right_blobs)
{
	int numInputs = left_blobs.size();

	if (numInputs != 1) 
	{
		print_err_str(LEFT_NOT_MATCH);
		exit(LEFT_NOT_MATCH);
	}

	int curr_idx;
	int x, y, out_ele;
	int i, j, k;
	int ii, jj;
	int in_x, in_y, in_z;
	int max_row, max_col, max_z;
	float curr_value;

	int batch_size = left_blobs[0]->batchSize;
	int ele_num = left_blobs[0]->get_ele_num();
	float *matrix_in = NULL, *matrix_out = NULL;
	coord_ptr tmp_coord = NULL;

	in_x = left_blobs[0]->x;
	in_y = left_blobs[0]->y;
	in_z = left_blobs[0]->z;

	x = right_blobs[0]->x;
	y = right_blobs[0]->y;
	out_ele = right_blobs[0]->get_ele_num();

	if (batch_size != right_blobs[0]->batchSize) return;

	tmp_space = (coordinate *)malloc(sizeof(coordinate) * out_ele * batch_size);
//	printf("batch_size = %d, ele_num = %d\n", batch_size, ele_num);
//	printf("in_x = %d, in_y = %d, in_z = %d, out_x = %d, out_y = %d\n", in_x, in_y, in_z, x, y);
//	printf("mask_x = %d, mask_y = %d, stride = %d\n", mask_x, mask_y, stride);
	for (curr_idx = 0; curr_idx < batch_size; curr_idx++)
	{
		matrix_in = left_blobs[0]->_data + curr_idx * ele_num / batch_size;
		matrix_out = right_blobs[0]->_data + curr_idx * out_ele / batch_size;
		tmp_coord = tmp_space + curr_idx * out_ele / batch_size;

//		printf("curr_idx = %d\n", curr_idx);
		for (k = 0; k < in_z; k++)
		{
			for (i = 0; i < in_y; i += stride)
			{
				for (j = 0; j < in_x; j += stride)
				{
					int inner_x = j / stride;
					int inner_y = i / stride;
					float curr_max = -1.0;

//					printf("inner_x = %d, inner_y = %d  ", inner_x, inner_y);
					for (ii = i; ii < i + mask_y; ii++)
					{
						for (jj = j; jj < j + mask_x; jj++)
						{
							if (ii < in_y && jj < in_x)
							{
								curr_value = matrix_in[(k * in_y + ii) * in_x + jj];
//								printf("curr_value = %f, ", curr_value);
							}
							else
							{
								curr_value = 0.0;
							}
							if (curr_max < curr_value)
							{
								curr_max = curr_value;
								max_row = ii;
								max_col = jj;
								max_z = k;
							}
//							printf("  curr_max = %f  ", curr_max);
						}
					}
//					printf("k = %d, y = %d, inner_y = %d, x = %d, inner_x = %d, sum = %d, \n",k, y, inner_y, x, inner_x, (k * y + inner_y) * x + inner_x);
					if (inner_x < x && inner_y < y)	
					{
//						printf("k = %d, y = %d, inner_y = %d, x = %d, inner_x = %d, sum = %d, curr_max = %f\n",k, y, inner_y, x, inner_x, (k * y + inner_y) * x + inner_x, curr_max);
						matrix_out[(k * y + inner_y) * x + inner_x] = curr_max;
						tmp_coord[(k * y + inner_y) * x + inner_x].row = max_row;
						tmp_coord[(k * y + inner_y) * x + inner_x].col = max_col;
						tmp_coord[(k * y + inner_y) * x + inner_x].z = max_z;
					}
//					printf("\n");
				}
			}
		}
	}

}
	

void PoolingLayer::bp(vector<Blob*> left_blobs, vector<Blob*> right_blobs)
{
	int numInputs = right_blobs.size();

	if (numInputs != 1) 
	{
		print_err_str(LEFT_NOT_MATCH);
		exit(LEFT_NOT_MATCH);
	}

	int curr_idx;
	int x, y, z, out_ele;
	int i, j, k;
	int ii, jj;
	int in_x, in_y, in_z;
	int max_row, max_col, max_z;
	float curr_value;

	int batch_size = right_blobs[0]->batchSize;
	int ele_num = right_blobs[0]->get_ele_num();
	float *matrix_in = NULL, *matrix_out = NULL;
	coord_ptr tmp_coord = NULL;

	in_x = right_blobs[0]->x;
	in_y = right_blobs[0]->y;
	in_z = right_blobs[0]->z;

	x = left_blobs[0]->x;
	y = left_blobs[0]->y;
	z = left_blobs[0]->z;

	out_ele = left_blobs[0]->get_ele_num();

	if (batch_size != right_blobs[0]->batchSize) return;

	for (curr_idx = 0; curr_idx < batch_size; curr_idx++)
	{
		matrix_in = right_blobs[0]->_data + curr_idx * ele_num / batch_size;
		matrix_out = left_blobs[0]->_data + curr_idx * out_ele / batch_size;
		tmp_coord = tmp_space + curr_idx * ele_num / batch_size;

		for (k = 0; k < z; k++)
		{
			for (i = 0; i < y; i++)
			{
				for (j = 0; j < x; j++)
				{
					matrix_out[(k * y + i) * x + j] = 0.0;
				}
			}
		}

		for (k = 0; k < in_z; k++)
		{
			for (i = 0; i < in_y; i++)
			{
				for (j = 0; j < in_x; j++)
				{
					int inner_x = tmp_coord[(k * in_y + i) * in_x + j].col;
					int inner_y = tmp_coord[(k * in_y + i) * in_x + j].row;
//					printf("recover batch_size = %d, row = %d, col = %d, layer = %d, sum = %d\n", curr_idx, inner_y, inner_x, k, (k*in_y +i)*in_x + j);
					matrix_out[(k * y + inner_y) * x + inner_x] = matrix_in[(k * in_y + i) * in_x + j];
				}
			}
		}
	}
	
}
	
void PoolingLayer::get_outputs_dimensions(int inputs_dims[], const int numInputs, int outputs_dims[], const int numOutputs)
{
	int i;
	for (i = 0; i < numOutputs; i++)
	{
		outputs_dims[i * 4 + 0] = inputs_dims[i * 4 + 0];
		outputs_dims[i * 4 + 1] = (inputs_dims[i * 4 + 1] - this->mask_x) / this->stride + 1;
		outputs_dims[i * 4 + 2] = (inputs_dims[i * 4 + 2] - this->mask_y) / this->stride + 1;
		outputs_dims[i * 4 + 3] = inputs_dims[i * 4 + 3];
	}
}
	
bool PoolingLayer::check_dimensions(){return true;}
