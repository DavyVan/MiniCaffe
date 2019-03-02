/***
 * @file net.h
 * @author Quan Fan
 * @brief Header file of Net class which represent the neural network itself.
 * @date 20/Feb/2019
 */

#ifndef _NET_H_
#define _NET_H_

#include <vector>
// #include "layer.h"   // Do not include this file due to cyclic including.
#include "blob.h"

using std::vector;

class Layer;

class Net
{
    public:
        Net();
        ~Net();

        /***
         * @brief Initialize neural network before running.
         *        Initialization includes allocation of memory, and reset all the blobs or variables to proper values.
         * 
         * @return The error code. 0 for normal.
         */
        int init();

        /***
         * @brief Training process consists of inference and BP.
         * 
         */
        void train();

        void bp();

        void infer();

        Blob get_output();  // tail

        /***
         * @brief Add new layer to this net.
         * This function will create necessary blobs to connect layers. It will firstly check the tail pointer by lefts' name and then search the net if not match.
         * If any designated lefts do not exist, function will report and abort. If any right blob has the same name as existed blob, report and abort.
         * 
         * @param layer     The layer to be added to this net.
         * @param lefts     The name set of left blobs.
         * @param rights    The name set of right blobs.
         * 
         * @return The error code.
         */
        int add_layer(Layer* layer, const char* lefts[], const char* rights[]);
    
    private:
        vector<Layer*> layers;
        vector<Blob*> blobs;
        
        vector<Layer*> genesis;
        vector<Blob*> tail;
        
};

#endif