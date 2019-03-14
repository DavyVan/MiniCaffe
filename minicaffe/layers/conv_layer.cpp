//
// Created by mingchen on 3/2/19.
//

#include "conv_layer.h"
#include <cstring>
ConvLayer::ConvLayer(char* name,int in_width, int in_height,int kernel_size, int in_channels, int out_channels,
          int w_stride,int h_stride):Layer(name),
          in_width(in_width),
          in_height(in_height),
          kernel_size(kernel_size),
          in_channels(in_channels),
          out_channels(out_channels),
          w_stride(w_stride),
          h_stride(h_stride)
          {
    weights=Blob("conv_weights",out_channels,kernel_size,kernel_size,in_channels);
    delta=Blob("conv_delta",out_channels,kernel_size,kernel_size,in_channels);
    weights.init();
    delta.init();
    out_width=(in_width-kernel_size)/w_stride+1;
    out_height=(in_height-kernel_size)/h_stride+1;
}

void ConvLayer::infer(std::vector<Blob*> lefts, std::vector<Blob*> rights){
    rights[0]->reset();
    for(int batch=0;batch<lefts[0]->batchSize;batch++){
        for(int InChannel=0;InChannel<in_channels;InChannel++){
            for(int OutChannel=0;OutChannel<out_channels;OutChannel++){
                for(int OutX=0;OutX<out_width;OutX++){
                    for(int OutY=0;OutY<out_height;OutY++){
                        for(int KernelX=0;KernelX<kernel_size;KernelX++){
                            for(int KernelY=0;KernelY<kernel_size;KernelY++){
                                int InX=w_stride*OutX+KernelX;
                                int InY=h_stride*OutY+KernelY;
                                float kernelMulti=weights(OutChannel,KernelX,KernelY,InChannel);
                                (*rights[0])(batch,OutX,OutY,OutChannel)+=
                                        (*lefts[0])(batch,InX,InY,InChannel)*kernelMulti;
                            }
                        }
                    }
                }
            }
        }
    }
}

void ConvLayer::get_outputs_dimensions(int *inputs_dims, const int numInputs, int *outputs_dims,
                                       const int numOutputs) {
    int temp[4]{inputs_dims[0],out_width,out_height,out_channels};
    memcpy(outputs_dims,&temp,4* sizeof(int));
}

bool ConvLayer::check_dimensions() {

}
void ConvLayer::bp(std::vector<Blob *> lefts, std::vector<Blob *> rights) {
    Blob input=*lefts[0];
    delta.reset();
    lefts[0]->reset();
    for(int batch=0;batch<lefts[0]->batchSize;batch++){
        for(int InChannel=0;InChannel<in_channels;InChannel++){
            for(int OutChannel=0;OutChannel<out_channels;OutChannel++){
                for(int OutX=0;OutX<out_width;OutX++){
                    for(int OutY=0;OutY<out_height;OutY++){
                        for(int KernelX=0;KernelX<kernel_size;KernelX++){
                            for(int KernelY=0;KernelY<kernel_size;KernelY++){
                                int InX=w_stride*OutX+KernelX;
                                int InY=h_stride*OutY+KernelY;
                                float rightGra=weights(batch,OutX,OutY,OutChannel);
                                delta(batch,KernelX,KernelY,InChannel)+=
                                        input(batch,InX,InY,InChannel)*rightGra;
                            }
                        }
                    }
                }
            }
        }
    }


}
int ConvLayer::init(){

}