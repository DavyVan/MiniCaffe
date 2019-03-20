/***
 * @file sigmoid.cpp
 * @author Quan Fan
 * @brief 
 * @date 19/Mar/2019
 */

#include "sigmoid.h"
#include <cmath>

SigmoidLayer::SigmoidLayer(char* name) : Layer(name){}

void SigmoidLayer::infer(std::vector<Blob*> lefts, std::vector<Blob*> rights)
{
    Blob* left = lefts[0];
    Blob* right = rights[0];
    int n = left->get_ele_num();
    for (int i = 0; i < n; i++)
        right->_data[i] = sigmoid(left->_data[i]);
}

void SigmoidLayer::bp(std::vector<Blob*> lefts, std::vector<Blob*> rights)
{
    printf("[Error] SigmoidLayer back propagation is not implemented.\n");
    exit(1);
}

void SigmoidLayer::get_outputs_dimensions(int inputs_dims[], const int numInputs, int outputs_dims[], const int numOutputs){}

bool SigmoidLayer::check_dimensions(){return true;}

int SigmoidLayer::init(){return 0;}

inline float SigmoidLayer::sigmoid(float x)
{
    return 0.5 * tanh(0.5 * x) + 0.5;
}