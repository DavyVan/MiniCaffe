#ifndef _PL_H_
#define _PL_H_

#include <cstdlib>
//#include "net.h"
#include "layer.h"
#include "errors.h"
#include "util.h"

class PoolingLayer : public Layer
{
public:
        int mask_x;
        int mask_y;
        int stride;
        coord_ptr tmp_space;

        PoolingLayer();
        PoolingLayer(char *name);
        PoolingLayer(char *name, int mask_x_, int mask_y_, int stride_);
        ~PoolingLayer();

        int init();
        void infer(vector<Blob*> left_blobs, vector<Blob*> right_blobs);
        void infer_gpu(vector<Blob*> left_blobs, vector<Blob*> right_blobs);
        void bp(vector<Blob*> left_blobs, vector<Blob*> right_blobs);
        void bp_gpu(vector<Blob*> left_blobs, vector<Blob*> right_blobs);
        void get_outputs_dimensions(int inputs_dims[], const int numInputs, int outputs_dims[], const int numOutputs);
        bool check_dimensions();
};

#endif
