/***
 * @file fc_layer.cpp
 * @author Quan Fan
 * @brief 
 * @date 17/Mar/2019
 */

#include "fc_layer.h"
#include "../seqnet.h"
#include "../util.h"
#include <cstdlib>

FCLayer::FCLayer(char* name, int num_output, bool bias_term, int flattened_dim) : Layer(name)
{
    N_ = num_output;
    M_ = SeqNet::get_batchsize();
    this->bias_term = bias_term;
    K_ = flattened_dim;
}

void FCLayer::get_outputs_dimensions(int inputs_dims[], const int numInputs, int outputs_dims[], const int numOutputs)
{
    // check
    if (numInputs != 1)
        printf("Erros: FCLayer needs one input not %d, the first one will be used.\n", numInputs);
    if (numOutputs != 1)
        printf("Erros: FCLayer needs one output not %d, the first one will be used.\n", numOutputs);
   
    int _batch = inputs_dims[0];
    
    if (K_ == 0)
    {
        int _x = inputs_dims[1];
        int _y = inputs_dims[2];
        int _z = inputs_dims[3];
        K_ = _x * _y * _z;
    }

    outputs_dims[0] = _batch;
    outputs_dims[1] = 1;
    outputs_dims[2] = N_;
    outputs_dims[3] = 1;
}

bool FCLayer::check_dimensions() {return true;}

int FCLayer::init()
{
    // weights (K x N)
    weight = new float[K_ * N_];
    random_matrix(weight, K_ * N_);

    // bias (N)
    bias = new float[N_];
    random_matrix(bias, N_);

    return 0;
}

void FCLayer::infer(std::vector<Blob*> lefts, std::vector<Blob*> rights)
{
    Blob* left = lefts[0];
    Blob* right = rights[0];

    // treat _data as a 2D matrix
    // right = left * weight
    simple_gemm(M_, N_, K_, 1, left->_data, weight, 0, right->_data);

    // bias
    if (bias_term)
    {
        for (int b = 0; b < M_; b++)
            vector_add(N_, &right->_data[b * N_], bias, &right->_data[b * N_]);
    }
}