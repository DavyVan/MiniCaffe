//
// Created by mingchen on 3/2/19.
//

#ifndef MAIN_CONV_LAYER_H
#define MAIN_CONV_LAYER_H

#include "../layer.h"
#include "../blob.h"
#include <string>
class ConvLayer: public Layer {
public:
    ConvLayer(char* name,int in_width, int in_height,int kernel_size, int in_channels, int out_channels,
        int w_stride,int h_stride);

    void infer(std::vector<Blob*> lefts, std::vector<Blob*> rights);
    void bp(std::vector<Blob*> lefts, std::vector<Blob*> rights);
    void infer_gpu(std::vector<Blob*> lefts, std::vector<Blob*> rights);
    void bp_gpu(std::vector<Blob*> lefts, std::vector<Blob*> rights);
    void get_outputs_dimensions(int inputs_dims[], const int numInputs, int outputs_dims[], const int numOutputs);
    bool check_dimensions();
    int init();

    // orgrinized as output_channel*kernel_x*kernel_y*input_channel
    Blob weights;

private:
    // orgrinized as output_channel*kernel_x*kernel_y*input_channel
    Blob delta;
    float* bias;
    float* delta_bias;
    int kernel_size;
    int in_width;
    int in_height;
    int in_channels;
    int out_channels;
    int w_stride;
    int h_stride;
    int out_width;
    int out_height;
    void update(int batchSize);
    void conv( Blob &input, Blob &kernel,Blob &output,int w_stride, int h_stride);
    void manage_mem_and_conv(Blob &input, Blob &kernel, Blob &output, int w_stride, int h_stride, float* bias=NULL);
    };


#endif //MAIN_CONV_LAYER_H
