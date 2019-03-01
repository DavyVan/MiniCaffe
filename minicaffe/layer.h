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
        char* name;                 /**< The name of this layer. */

        vector<Blob*> left_blobs;         /**< Multiple input/output blobs. Left means input for inference and output for bp. */
        vector<Blob*> right_blobs;        /**< Multiple output/output blobs. Right means output for inference and input for bp.*/

        virtual int init()=0;

        virtual void infer()=0;

        virtual void bp()=0;

        /***
         * @brief Calculate the outputs' dimensions given inputs' dimensions.
         * This function is used by @Net to construct Blobs, and layer itself.
         * 
         * @param intputs_dims [in] Dimensions of input blobs. [batchSize0, x0, y0, z0, batchSize1, x1, y1, z1, ...]
         * @param outputs_dims [out] Dimensions of output blobs. It has the same format as @inputs_dims
         */
        virtual void get_outputs_dimension(int inputs_dims[], int outputs_dims[])=0;

    protected:

        /***
         * @brief Constructor of Layers' base class.
         * Accept layer's name.
         */
        Layer(char* name);

        /***
         * @brief Add itself to @Net
         * It is implemented as a member function to ensure that a instance must be constructed before we can add a layer to net.
         * It is designed to be called from wrapper.
         * 
         * @param net       The model user created.
         * @param lefts     Blobs on the left. Instantiation of Blobs will take place in @Net so we only need string identifiers.
         *                  Actually, lefts are already constructed when adding the previous layer.
         * @param rights    Blobs on the right. Instantiation of Blobs will take place in @Net so we only need string identifiers.
         * 
         * @return Error code.
         */
        int add_to_net(Net* net, const char* lefts[], const char* rights[]);
};

#endif