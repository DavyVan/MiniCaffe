//
// Created by mingchen on 3/2/19.
//

#include "conv_layer.h"
#include <cstring>
#include <random>
#include "../util.h"
ConvLayer::ConvLayer(char *name, int in_width, int in_height, int kernel_size, int in_channels, int out_channels,
                     int w_stride, int h_stride) : Layer(name),
                                                   in_width(in_width),
                                                   in_height(in_height),
                                                   kernel_size(kernel_size),
                                                   in_channels(in_channels),
                                                   out_channels(out_channels),
                                                   w_stride(w_stride),
                                                   h_stride(h_stride) {
    weights = Blob("conv_weights", out_channels, kernel_size, kernel_size, in_channels);
    delta = Blob("conv_delta", out_channels, kernel_size, kernel_size, in_channels);
    weights.init();
    delta.init();
    bias = new float[out_channels];
    delta_bias = new float[out_channels];
    out_width = (in_width - kernel_size) / w_stride + 1;
    out_height = (in_height - kernel_size) / h_stride + 1;
    std::default_random_engine e;
    std::uniform_real_distribution<float> ran(0.0001, 1.0);
    for (int OutChannel = 0; OutChannel < out_channels; OutChannel++) {
        bias[OutChannel] = ran(e);
        for (int InChannel = 0; InChannel < in_channels; InChannel++) {
            for (int KernelX = 0; KernelX < kernel_size; KernelX++) {
                for (int KernelY = 0; KernelY < kernel_size; KernelY++) {
                    weights(OutChannel, KernelX, KernelY, InChannel) = ran(e);
                }
            }
        }
    }
}




void ConvLayer::infer(std::vector<Blob *> lefts, std::vector<Blob *> rights) {
    rights[0]->reset();
    conv(*lefts[0], weights, *rights[0], w_stride, h_stride);
    for (int batch = 0; batch < lefts[0]->batchSize; batch++) {
        for (int OutChannel = 0; OutChannel < out_channels; OutChannel++) {
            for (int OutX = 0; OutX < out_width; OutX++) {
                for (int OutY = 0; OutY < out_height; OutY++) {
                    (*rights[0])(batch, OutX, OutY, OutChannel) += bias[OutChannel];
                }
            }
        }
    }
}

void ConvLayer::get_outputs_dimensions(int *inputs_dims, const int numInputs, int *outputs_dims,
                                       const int numOutputs) {
    int temp[4]{inputs_dims[0], out_width, out_height, out_channels};
    memcpy(outputs_dims, &temp, 4 * sizeof(int));
}

bool ConvLayer::check_dimensions() {
    return true;
}

void ConvLayer::bp(std::vector<Blob *> lefts, std::vector<Blob *> rights) {
    Blob input = *lefts[0];
    delta.reset();
    lefts[0]->reset();
    memset(delta_bias, 0, sizeof(float));
    for (int batch = 0; batch < lefts[0]->batchSize; batch++) {
        for (int InChannel = 0; InChannel < in_channels; InChannel++) {
            for (int OutChannel = 0; OutChannel < out_channels; OutChannel++) {
                for (int OutX = 0; OutX < out_width; OutX++) {
                    for (int OutY = 0; OutY < out_height; OutY++) {
                        for (int KernelX = 0; KernelX < kernel_size; KernelX++) {
                            for (int KernelY = 0; KernelY < kernel_size; KernelY++) {
                                int InX = w_stride * OutX + KernelX;
                                int InY = h_stride * OutY + KernelY;
                                float rightGra = (*rights[0])(batch, OutX, OutY, OutChannel);
                                delta(OutChannel, KernelX, KernelY, InChannel) +=
                                        input(batch, InX, InY, InChannel) * rightGra;
                            }
                        }
                    }
                }
            }
        }
    }
    for (int batch = 0; batch < lefts[0]->batchSize; batch++) {
        for (int OutChannel = 0; OutChannel < out_channels; OutChannel++) {
            for (int OutX = 0; OutX < out_width; OutX++) {
                for (int OutY = 0; OutY < out_height; OutY++) {
                    delta_bias[OutChannel] += (*rights[0])(batch, OutX, OutY, OutChannel);
                }
            }
        }
    }

    Blob warped_right("warped_right", rights[0]->batchSize, 2 * kernel_size + (w_stride) * (out_width-1) - 1,
                      2 * kernel_size + (h_stride) * (out_height-1) - 1, out_channels);
    warped_right.init();
    Blob warped_kernel("warped_kernel", in_channels, kernel_size, kernel_size, out_channels);
    warped_kernel.init();
    for (int InChannel = 0; InChannel < in_channels; InChannel++) {
        for (int OutChannel = 0; OutChannel < out_channels; OutChannel++) {
            for (int KernelX = 0; KernelX < kernel_size; KernelX++) {
                for (int KernelY = 0; KernelY < kernel_size; KernelY++) {
                    warped_kernel(InChannel, KernelX, KernelY, OutChannel) =
                            weights(OutChannel, kernel_size - KernelX-1, kernel_size - KernelY-1, InChannel);
                }
            }
        }
    }
    for (int batch = 0; batch < lefts[0]->batchSize; batch++) {
        for (int OutChannel = 0; OutChannel < out_channels; OutChannel++) {
            for (int OutX = 0; OutX < out_width; OutX++) {
                for (int OutY = 0; OutY < out_height; OutY++) {
                    warped_right(batch,(kernel_size-1)+OutX*w_stride,(kernel_size-1)+OutY*h_stride,OutChannel)=(*rights[0])(batch,OutX,OutY,OutChannel);
                }
            }
        }
    }
    conv(warped_right,warped_kernel,*lefts[0],1,1);
    update(lefts[0]->batchSize);
}



void ConvLayer::conv(Blob &input, Blob &kernel, Blob &output, int w_stride, int h_stride) {
    int batch_size = input.batchSize;
    int in_width = input.x;
    int in_height = input.y;
    int kernel_size = kernel.x;
    int in_channels = input.z;
    int out_channels = kernel.batchSize;
    int out_width = (in_width - kernel_size) / w_stride + 1;
    int out_height = (in_height - kernel_size) / h_stride + 1;

    for (int batch = 0; batch < batch_size; batch++) {
        for (int InChannel = 0; InChannel < in_channels; InChannel++) {
            for (int OutChannel = 0; OutChannel < out_channels; OutChannel++) {
                for (int OutX = 0; OutX < out_width; OutX++) {
                    for (int OutY = 0; OutY < out_height; OutY++) {
                        for (int KernelX = 0; KernelX < kernel_size; KernelX++) {
                            for (int KernelY = 0; KernelY < kernel_size; KernelY++) {
                                int InX = w_stride * OutX + KernelX;
                                int InY = h_stride * OutY + KernelY;
                                float kernelMulti = kernel(OutChannel, KernelX, KernelY, InChannel);
                                output(batch, OutX, OutY, OutChannel) +=
                                        input(batch, InX, InY, InChannel) * kernelMulti;
                            }
                        }
                    }
                }
            }
        }
    }
}

void ConvLayer::update(int batchSize) {
    for (int InChannel = 0; InChannel < in_channels; InChannel++) {
        for (int OutChannel = 0; OutChannel < out_channels; OutChannel++) {
            for (int KernelX = 0; KernelX < kernel_size; KernelX++) {
                for (int KernelY = 0; KernelY < kernel_size; KernelY++) {
                    weights(OutChannel, KernelX, KernelY, InChannel) +=
                            delta(OutChannel, KernelX, KernelY, InChannel) / float(batchSize);
                }
            }
        }
    }
    for (int OutChannel = 0; OutChannel < out_channels; OutChannel++) {
        bias[OutChannel] += delta_bias[OutChannel] / float(batchSize);
    }

}

int ConvLayer::init() {
    return 0;
}