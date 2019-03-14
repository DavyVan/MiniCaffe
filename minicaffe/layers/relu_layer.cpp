//
// Created by mingchen on 3/2/19.
//

#include "relu_layer.h"
#include <math.h>
void ReluLayer::infer(vector<Blob *> lefts, vector<Blob *> rights) {
    for (int i=0;i<lefts[0]->get_ele_num();i++)
        rights[0]->_data[i]=(float)0.0>lefts[0]->_data[i]?(float)0.0:lefts[0]->_data[i];
}

void ReluLayer::bp(vector<Blob *> lefts, vector<Blob *> rights) {
    for(int i=0;i<rights[0]->get_ele_num();i++){
        lefts[i]->_data[i]=(float)0.0>rights[0]->_data[i]?(float)0.0:rights[0]->_data[i];
    }
}
bool ReluLayer::check_dimensions() {
    return 1;
}
void ReluLayer::get_outputs_dimensions(int *inputs_dims, const int numInputs, int *outputs_dims,
                                       const int numOutputs) {
    memcpy(outputs_dims,inputs_dims,4*numOutputs*sizeof(int));
}
int ReluLayer::init() {}