/***
 * @file net.h
 * @author Quan Fan
 * @brief Header file of Net class which represent the neural network itself.
 * @date 20/Feb/2019
 */

#ifndef _NET_H_
#define _NET_H_

#include "layer.h"
#include "blob.h"

class Net
{
    public:
        Net();
        ~Net();

        /***
         * @brief Initialize neural network after adding layers.
         *        Initialization includes allocation of memory, and reset all the blobs or variables to proper values.
         * 
         * @return int The error number. 0 for normal.
         */
        int init();

        /***
         * @brief Training process consists of inference and BP.
         * 
         */
        void train();

        void bp();

        void infer();

        Blob get_output();

        int add_layer(Layer* layer);
    
    private:
        //TODO: data structure to store the network
        //TODO: data structure to store the blobs
};

#endif