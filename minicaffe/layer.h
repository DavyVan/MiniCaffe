/***
 * @file layer.h
 * @author Quan Fan
 * @brief Define the abstract base class of all layers.
 * @date 20/Feb/2019
 */

#ifndef _LAYER_H_
#define _LAYER_H_

/***
 * @brief The common abstract base class of all layers, which only contains the most basic attributes about a layer.
 * 
 */
class Layer
{
    public:
        char* name; /**< The name of this layer. */

        virtual int init()=0;

        virtual void infer()=0;

        virtual void bp()=0;

    protected:
        Layer();
};

#endif