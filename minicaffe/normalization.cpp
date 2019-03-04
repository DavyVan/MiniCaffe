#include "normalization.h"
#include <math.h>

NormalizationLayer::NormalizationLayer():Layer("undefined")
{
	nn = 5;
	kk = 2.0;
	alpha = 1e-4;
	beta = 0.75;
}

NormalizationLayer::NormalizationLayer(char *name):Layer(name)
{

	nn = 5;
	kk = 2.0;
	alpha = 1e-4;
	beta = 0.75;
}

NormalizationLayer::NormalizationLayer(char *name, float alpha_, float beta_, float kk_, int nn_):Layer(name)
{
	nn = nn_;
	kk = kk_;
	alpha = alpha_;
	beta = beta_;
}

NormalizationLayer::~NormalizationLayer()
{
	;
}

int NormalizationLayer::init() {return 0;}

void NormalizationLayer::infer()
{
	int curr_idx;
	int i, j, k, t;
	int ii, jj;
	int in_x, in_y, in_z;
	int idx;
	int numInputs = left_blobs.size();
	float curr_value;

	if (numInputs != 1) 
	{
		print_err_str(LEFT_NOT_MATCH);
		exit(LEFT_NOT_MATCH);
	}

	int batch_size = left_blobs[0]->batchSize;
	int ele_num = left_blobs[0]->get_ele_num();

	for (curr_idx = 0; curr_idx < batch_size; curr_idx++)
	{
		float *matrix_in = left_blobs[0]->_data + batch_size * ele_num;
		float *matrix_out = right_blobs[0]->_data + batch_size * ele_num;

		in_x = left_blobs[0]->x;
		in_y = left_blobs[0]->y;
		in_z = left_blobs[0]->z;

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

void NormalizationLayer::bp()
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

void NormalizationLayer::get_outputs_dimensions(int inputs_dims[], const int numInputs, int outputs_dims[], const int numOutputs)
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

bool NormalizationLayer::check_dimensions(){return true;}



