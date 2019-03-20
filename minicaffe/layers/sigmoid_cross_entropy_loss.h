/***
 * @file sigmoid_cross_entropy_loss.h
 * @author Quan Fan
 * @brief 
 * @date 19/Mar/2019
 */

#ifndef _SIGMOID_CROSS_ENTROPY_LOSS_H_
#define _SIGMOID_CROSS_ENTROPY_LOSS_H_

#include "../layer.h"
#include "sigmoid.h"

class SigmoidCrossEntropyLoss : public Layer
{
public:
    SigmoidCrossEntropyLoss(char* name);
    void infer(std::vector<Blob*> lefts, std::vector<Blob*> rights);
    void bp(std::vector<Blob*> lefts, std::vector<Blob*> rights);
    void infer_gpu(std::vector<Blob*> lefts, std::vector<Blob*> rights);
    void bp_gpu(std::vector<Blob*> lefts, std::vector<Blob*> rights);
    void get_outputs_dimensions(int inputs_dims[], const int numInputs, int outputs_dims[], const int numOutputs); 
    bool check_dimensions();
    int init();
private:
    SigmoidLayer sigmoidLayer;
    Blob* sigmoid_output;
};

#endif