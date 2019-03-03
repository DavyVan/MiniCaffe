#include "net.h"
#include "layer.h"
#include "errors.h"

class PoolingLayer : public Layer
{
private:
	int mask_x, mask_y, stride;

public:
	PoolingLayer()
	{
		mask_x = 5;
		mask_y = 5;
		stride = 2;
	}

	PoolingLayer(int x, int y, int s)
	{
		mask_x = x;
		mask_y = y;
		stride = s;
	}

	~PoolingLayer(){;}
	
	int init() {return 0;}

	void infer()
	{
		int curr_idx;
		int i, j, k;
		int ii, jj;
		int in_x, in_y, in_z;
		float curr_value;

		for (curr_idx = 0; curr_idx < numInputs; curr_idx++)
		{
			float *matrix_in = left_blobs[curr_idx]._data;
			float *matrix_out = right_blobs[curr_idx]._data;

			in_x = inputs_dims[curr_idx * 4 + 1];
			in_y = inputs_dims[curr_idx * 4 + 2];

			for (k = 0; k < in_z; k++)
			{
				for (i = 0; i < in_y; i += stride)
				{
					for (j = 0; j < in_x; j += stride)
					{
						int inner_x = j / stride;
						int inner_y = i / stride;
						float curr_max = -1.0;
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
							}
						}
						matrix_out[(k * y + inner_y) * x + inner_x] = curr_max;
					}
				}
			}
		}
	}

	void bp()
	{
		int i;
		int curr_idx;
		for (curr_idx = 0; curr_idx < numInputs; curr_idx++)
		{
			int ele_num = left_blobs[curr_idx].get_ele_num();
			float *matrix_in = left_blobs[curr_idx]._data;
			float *matrix_out = right_blobs[curr_idx]._data;
			for (i = 0; i < ele_num; i++)
			{
				matrix_out[i] = matrix_in[i];
			}
		}
	}

	void get_outputs_dimensions(int inputs_dims[], const int numInputs, int outputs_dims[], const int numOutputs)
	{
		int i;
		for (i = 0; i < numOutputs; i++)
		{
			outputs_dims[i * 4 + 0] = inputs_dims[i * 4 + 0];
			outputs_dims[i * 4 + 1] = ( inputs_dims[i * 4 + 1] - mask_x ) / stride + 1;
			outputs_dims[i * 4 + 2] = ( inputs_dims[i * 4 + 2] - mask_y ) / stride + 1;
			outputs_dims[i * 4 + 3] = inputs_dims[i * 4 + 3];
		}
	}

	PoolingLayer(char* name):Layer(name){}

	bool check_dimensions(){return true;}
}
