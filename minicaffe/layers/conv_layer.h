//
// Created by mingchen on 3/2/19.
//

#ifndef MAIN_CONV_LAYER_H
#define MAIN_CONV_LAYER_H

#include "../layer.h"
#include "../blob.h"
#include <string>
class ConvLayer:Layer {
public:
    ConvLayer(char* name,int in_width, int in_height,int kernel_size, int in_channels, int out_channels,
        int w_stride,int h_stride);

    void infer(std::vector<Blob*> lefts, std::vector<Blob*> rights);
    void bp(std::vector<Blob*> lefts, std::vector<Blob*> rights);
    void get_outputs_dimensions(int inputs_dims[], const int numInputs, int *&outputs_dims, const int numOutputs);
    bool check_dimensions();
    int init();
    Blob weights;

private:

    Blob delta;
    int kernel_size;
    int in_width;
    int in_height;
    int in_channels;
    int out_channels;
    int w_stride;
    int h_stride;
    int out_width;
    int out_height;
};


#endif //MAIN_CONV_LAYER_H
