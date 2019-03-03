#include "net.h"
#include "layer.h"
#include "errors.h"

class  NormalizationLayer : public layer
{
private:
	int nn;
	float alpha, beta, kk;

public:
	NormalizationLayer(char *name)
	{
		nn = 5;
		kk = 2.0;
		alpha = 1e-4;
		beta = 0.75;
		Layer(name);
	}

	NormalizationLayer(char *name, float alpha_, float beta_, float kk_, int nn_)
	{
		nn = nn_;
		kk = kk_;
		alpha = alpha_;
		beta = beta_;
		Layer(name);
	}

	~NormalizationLayer(){;}

	int init() {return 0;}

	void infer()
	{
		int curr_idx;
		int i, j, k, t;
		int ii, jj;
		int in_x, in_y, in_z;
		int idx;
		float curr_value;

		for (curr_idx = 0; curr_idx < numInputs; curr_idx++)
		{
			float *matrix_in = left_blobs[curr_idx]._data;
			float *matrix_out = right_blobs[curr_idx]._data;

			in_x = inputs_dims[curr_idx * 4 + 1];
			in_y = inputs_dims[curr_idx * 4 + 2];
			in_z = inputs_dims[curr_idx * 4 + 3];

			for (k = 0; k < in_z; k++)
			{
				for (i = 0; i < in_y; i++)
				{
					for (j = 0; j < in_x; j++)
					{
						double curr_sum = 0.0;
						int start = ( (k - nn / 2) < 0 ) ? 0 : (k - nn / 2);
						int end = ( (in_z - 1) < (k + nn / 2) ) ? (in_z - 1) : (k + nn / 2);
						for (t = start; t <= end; t++) 
						{
							curr_sum += matrix_in[(t * in_y + i) * in_x + j] * matrix_in[(t * in_y + i) * in_x + j];
						}

						idx = (k * in_y + i) * in_x + j;
						double inner_value = kk + alpha * curr_sum;
						matrix_out[idx] = matrix_in[idx] / pow(inner_value, beta);
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
			outputs_dims[i * 4 + 1] = inputs_dims[i * 4 + 1];
			outputs_dims[i * 4 + 2] = inputs_dims[i * 4 + 2];
			outputs_dims[i * 4 + 3] = inputs_dims[i * 4 + 3];
		}
	}

	bool check_dimensions(){return true;}
}
