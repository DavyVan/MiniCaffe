//
// Created by mingchen on 3/2/19.
//

#ifndef MAIN_RELU_LAYER_H
#define MAIN_RELU_LAYER_H

#include "../layer.h"
#include "../blob.h"
#include <string>

class ReluLayer:Layer {
public:
    ReluLayer(char* name):Layer(name){};

    void infer(vector<Blob*> lefts, vector<Blob*> rights);
    void bp(vector<Blob*> lefts, vector<Blob*> rights);
    void get_outputs_dimensions(int inputs_dims[], const int numInputs, int outputs_dims[], const int numOutputs);
    bool check_dimensions();
    int init();
    void infer_gpu(vector<Blob*> lefts, vector<Blob*> rights);
    void bp_gpu(vector<Blob*> lefts, vector<Blob*> rights);
};


#endif //MAIN_RELU_LAYER_H
