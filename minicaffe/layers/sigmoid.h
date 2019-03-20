/***
 * @file sigmoid.h
 * @author Quan Fan
 * @brief 
 * @date 19/Mar/2019
 */

#ifndef _SIGMOID_H_
#define _SIGMOID_H_

#include "../layer.h"

class SigmoidLayer : public Layer
{
public:
    SigmoidLayer(char* name);
    void infer(std::vector<Blob*> lefts, std::vector<Blob*> rights);
    void bp(std::vector<Blob*> lefts, std::vector<Blob*> rights);
    void get_outputs_dimensions(int inputs_dims[], const int numInputs, int outputs_dims[], const int numOutputs); 
    bool check_dimensions();
    int init();
    inline float sigmoid(float x);
};

#endif