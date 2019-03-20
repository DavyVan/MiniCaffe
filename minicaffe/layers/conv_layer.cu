#include "conv_layer.h"
__global__ void conv(float *out,float *in, unsigned size){
    
}


void ConvLayer::infer_gpu(std::vector<Blob *> lefts, std::vector<Blob *> rights) {
    infer(lefts, rights);
}

void ConvLayer::bp_gpu(std::vector<Blob *> lefts, std::vector<Blob *> rights) {
    bp(lefts, rights);
}
