/***
 * @file fc_layer.h
 * @author Quan Fan
 * @brief Fully connected layer (inner product layer)
 * @date 16/Mar/2019
 */

#ifndef _FC_LAYER_H_
#define _FC_LAYER_H_

#include "../layer.h"
#include "../blob.h"

class FCLayer : public Layer
{
public:
    FCLayer(char* name, int num_output, bool bias_term, int flattened_dim=0);
    void infer(std::vector<Blob*> lefts, std::vector<Blob*> rights);
    void bp(std::vector<Blob*> lefts, std::vector<Blob*> rights);
    void get_outputs_dimensions(int inputs_dims[], const int numInputs, int outputs_dims[], const int numOutputs);
    bool check_dimensions();
    int init(); // weights, bias
    void infer_gpu(std::vector<Blob*> left_blobs, std::vector<Blob*> right_blobs);
    void bp_gpu(std::vector<Blob*> left_blobs, std::vector<Blob*> right_blobs);
// private:
    int N_;         // output dim
    int K_;         // flattened dim
    int M_;         // batch size
    bool bias_term;
    float *weight = NULL;
    float *bias = NULL;
    // axis = 1;
    // weight & bias are filled randomly [0, 1]
};

#endif