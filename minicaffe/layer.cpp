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

int Layer::add_to_net(Net* net, const char* lefts[], const char* rights[])
{
    return net->add_layer(this, lefts, rights);
}