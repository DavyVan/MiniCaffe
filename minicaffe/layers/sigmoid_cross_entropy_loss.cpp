/***
 * @file sigmoid_cross_entropy_loss.cpp
 * @author Quan Fan
 * @brief 
 * @date 19/Mar/2019
 */

#include "sigmoid_cross_entropy_loss.h"
#include "../seqnet.h"
#include <cmath>

SigmoidCrossEntropyLoss::SigmoidCrossEntropyLoss(char* name) : Layer(name), sigmoidLayer("_sigmoid")
{
    sigmoid_output = NULL;
}

void SigmoidCrossEntropyLoss::infer(std::vector<Blob*> lefts, std::vector<Blob*> rights)
{
    // Call sigmoid layer
    sigmoidLayer.infer(lefts, std::vector<Blob*>({sigmoid_output}));

    Blob* input_data = lefts[0];
    Blob* target = lefts[1];
    float loss = 0;
    int n = input_data->get_ele_num();
    for (int i = 0; i < n; i++)
    {
        loss -= input_data->_data[i] * (target->_data[i] - (input_data->_data[i] >= 0 ? 1 : 0))
                - log(1 + exp(input_data->_data[i] - 2 * input_data->_data[i] * (input_data->_data[i] >= 0 ? 1 : 0)));
    }

    // use batch size as normalizer
    rights[0]->_data[0] = loss / SeqNet::get_batchsize();
}

void SigmoidCrossEntropyLoss::get_outputs_dimensions(int inputs_dims[], const int numInputs, int outputs_dims[], const int numOutputs)
{
    // equal to label's dimensions
    if (numInputs != 2)
    {
        printf("[Error] Sigmoid cross entropy loss layer needs 2 inputs rather than %d.\n", numInputs);
        exit(1);
    }
    outputs_dims[0] = 1;
    outputs_dims[1] = 1;
    outputs_dims[2] = 1;
    outputs_dims[3] = 1;
    sigmoid_output = new Blob("sigmoid_output", outputs_dims[0], outputs_dims[1], outputs_dims[2], outputs_dims[3]);
}

bool SigmoidCrossEntropyLoss::check_dimensions() {return true;}

int SigmoidCrossEntropyLoss::init()
{
    if (sigmoid_output == NULL)
    {
        printf("[Error] sigmoid_output was not initialized.\n");
        exit(1);
    }
    else
    {
        sigmoid_output->init();
    }
    return 0;
}