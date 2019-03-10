#include "pooling.h"
#include <math.h>
	
PoolingLayer::PoolingLayer():Layer("undefined")
{
	mask_x = 5;
	mask_y = 5;
	stride = 2;
}

PoolingLayer::PoolingLayer(char *name):Layer(name)
{
	mask_x = 5;
	mask_y = 5;
	stride = 2;
}

PoolingLayer::PoolingLayer(char *name, int mask_x_, int mask_y_, int stride_):Layer(name)
{
	mask_x = mask_x_;
	mask_y = mask_y_;
	stride = stride_;
}

PoolingLayer::~PoolingLayer()
{
	;
}

int PoolingLayer::init() {return 0;}

void PoolingLayer::infer()
{
	int numInputs = left_blobs.size();
	
	if (numInputs != 1) 
	{
		print_err_str(LEFT_NOT_MATCH);
		exit(LEFT_NOT_MATCH);
	}

	int curr_idx;
	int x, y;
	int i, j, k;
	int ii, jj;
	int in_x, in_y, in_z;
	float curr_value;

	int batch_size = left_blobs[0]->batchSize;
	int ele_num = left_blobs[0]->get_ele_num();

	for (curr_idx = 0; curr_idx < batch_size; curr_idx++)
	{
		float *matrix_in = left_blobs[0]->_data + batch_size * ele_num;
		float *matrix_out = right_blobs[0]->_data + batch_size * ele_num;
		in_x = left_blobs[0]->x;
		in_y = left_blobs[0]->y;
		in_z = left_blobs[0]->z;

		x = right_blobs[0]->x;
		y = right_blobs[0]->y;

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
					/* inner kernel, pooling operation */
					for (ii = i; ii < i + mask_y; ii++)
					{
						for (jj = j; jj < j + mask_x; jj++)
						{
							if (ii < in_y && jj < in_x)
							{
								curr_value = matrix_in[(k * in_y + ii) * in_x + jj];
							}
							else
							{
								curr_value = 0.0;
							}
							curr_max = (curr_max < curr_value) ? curr_value : curr_max;
//							printf("  curr_max = %f  ", curr_max);						}
						}
					}
					matrix_out[(k * y + inner_y) * x + inner_x] = curr_max;
//					printf("\n");
				}
			}
		}
	}

}
	

void PoolingLayer::bp()
{
	int i;
	int curr_idx;
	int numInputs = left_blobs.size();
	float curr_value;
	
	if (numInputs != 1) 
	{
		print_err_str(LEFT_NOT_MATCH);
		exit(LEFT_NOT_MATCH);
	}
	
	int batch_size = left_blobs[0]->batchSize;
	int ele_num = left_blobs[0]->get_ele_num();
	
	
	float *matrix_in = left_blobs[0]->_data;
	float *matrix_out = right_blobs[0]->_data;
	
	for (i = 0; i < ele_num; i++)
	{
		matrix_out[i] = matrix_in[i];
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
