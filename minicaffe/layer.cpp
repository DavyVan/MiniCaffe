/***
 * @file layer.cpp
 * @author Quan Fan
 * @brief Implementation of some essential function of Layer class.
 * @date 01/Mar/2019
 */

#include <cstring>
#include "layer.h"
#include "util.h"

Layer::Layer(char* name)
{
    alloc_and_strcpy(&(this->name), name);
}

int Layer::add_to_net(Net* net, const char* lefts[], const int numLefts, const char* rights[], const int numRights)
{
    return net->add_layer(this, lefts, numLefts, rights, numRights);
}

void Layer::infer_gpu(vector<Blob*> left_blobs, vector<Blob*> right_blobs)
{
    printf("infer_gpu in base class layer is not implemented.\n");
    exit(1);
}

void Layer::bp_gpu(vector<Blob*> lefts, vector<Blob*> rights)
{
    printf("infer_gpu in base class layer is not implemented.\n");
    exit(1);
}