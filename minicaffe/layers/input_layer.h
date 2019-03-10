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

    void infer();

};


#endif //MAIN_INPUT_LAYER_H
