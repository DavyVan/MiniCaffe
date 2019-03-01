/***
 * @file layer.h
 * @author Quan Fan
 * @brief Define the abstract base class of all layers.
 * @date 20/Feb/2019
 */

#ifndef _LAYER_H_
#define _LAYER_H_

#include <vector>
#include "net.h"
#include "blob.h"

using namespace std;

/***
 * @brief The common abstract base class of all layers, which only contains the most basic attributes about a layer.
 * 
 */
class Layer
{
    public:
        const char* name;                 /**< The name of this layer. */

        vector<Blob*> left_blobs;         /**< Multiple input/output blobs. Left means input for inference and output for bp. */
        vector<Blob*> right_blobs;        /**< Multiple output/output blobs. Right means output for inference and input for bp.*/

        virtual int init()=0;

        virtual void infer()=0;

        virtual void bp()=0;

    protected:
        Layer();    //TODO: init name

        int add_to_net(Net* net, const char* lefts[], const char* rights[]);   //TODO: call net->add_layer(this, lefts, rights)
};

#endif