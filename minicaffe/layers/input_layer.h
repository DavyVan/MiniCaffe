/***
 * @file input_layer.h
 * @author Mingchen Li
 * @brief Header file of input_layer
 * @date 02/Mar/2019
 */

#ifndef MAIN_INPUT_LAYER_H
#define MAIN_INPUT_LAYER_H

#include "../layer.h"
#include "../blob.h"
class InputLayer : public Layer
{
public:
    InputLayer(char* name);

    void infer(vector<Blob*> lefts, vector<Blob*> rights);
    void bp(vector<Blob*> lefts, vector<Blob*> rights);
    void infer_gpu(vector<Blob*> lefts, vector<Blob*> rights);
    void bp_gpu(vector<Blob*> lefts, vector<Blob*> rights);
    void get_outputs_dimensions(int inputs_dims[], const int numInputs, int outputs_dims[], const int numOutputs);
    bool check_dimensions();
    int init();
    void infer_gpu(vector<Blob*> lefts, vector<Blob*> rights);
    void bp_gpu(vector<Blob*> lefts, vector<Blob*> rights);
};


#endif //MAIN_INPUT_LAYER_H
