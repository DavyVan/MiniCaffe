#include "conv_layer.h"
#include <assert.h>
__global__ void
conv_gpu_stride(float *out, float *in, float *kernel, int in_width,
         int in_height,int out_width,int out_height, int kernel_size, int in_channels, int out_channels,
         int w_stride, int h_stride,float* bias) {
    int OutX=blockIdx.x*blockDim.x+threadIdx.x;
    int OutY=blockIdx.y*blockDim.y+threadIdx.y;
    int OutChannel=blockIdx.z*blockDim.z+threadIdx.z;
    int out_pos=OutY*out_width*out_channels+OutX*out_channels+OutChannel;
    if(OutX<out_width && OutY<out_height){
        for(int KernelX=0;KernelX<kernel_size;KernelX++){
            for(int KerenlY=0;KerenlY<kernel_size;KerenlY++){
                for(int InChannel=0;InChannel<in_channels;InChannel++){
                    int InX=OutX*w_stride+KernelX;
                    int InY=OutY*h_stride+KerenlY;
                    int in_pos=InY*in_width*in_channels+InX*in_channels+InChannel;
                    int kernel_pos=OutChannel*kernel_size*kernel_size*in_channels
                                   +KerenlY*kernel_size*in_channels
                                   +KernelX*in_channels
                                   +InChannel;

                    out[out_pos]+=in[in_pos]*kernel[kernel_pos];
                }

            }
        }
        out[out_pos]+=bias[OutChannel];
    }
}

__global__ void
conv_gpu_dilation(float *out, float *in, float *kernel, int in_width,
                  int in_height,int out_width,int out_height, int kernel_size, int in_channels, int out_channels,
                  int w_dilation, int h_dilation,float* bias){

}

void ConvLayer::infer_gpu(std::vector<Blob *> lefts, std::vector<Blob *> rights) {

    manage_mem_and_conv(*lefts[0],weights,*rights[0],w_stride,h_stride,bias);
//    rights[0]->reset();
//    cudaError_t cuda_ret;
//    dim3 dim_grid,dim_block;
//    dim_block.x=16; dim_block.y=16; dim_block.z=1;
//    dim_grid.x=(out_width-1)/16+1; dim_grid.y=(out_width-1)/16+1; dim_grid.z=out_channels;
//
//    float *in_h,*in_d,*kernel_h,*kernel_d,*out_d,*bias_d;
//
//    // device memory alloc
//    in_h=lefts[0]->_data;
//    kernel_h=weights._data;
//    cuda_ret = cudaMalloc((void**)&in_d, lefts[0]->get_ele_num() * sizeof(float));
//    assert(cuda_ret == cudaSuccess);
//
//    cuda_ret = cudaMalloc((void**)&kernel_d, weights.get_ele_num() * sizeof(float));
//    assert(cuda_ret == cudaSuccess);
//
//    cuda_ret = cudaMalloc((void**)&out_d, rights[0]->get_ele_num() * sizeof(float));
//    assert(cuda_ret == cudaSuccess);
//
//    cuda_ret = cudaMalloc((void**)&bias_d, out_channels * sizeof(float));
//    assert(cuda_ret == cudaSuccess);
//
//    cudaDeviceSynchronize();
//
//    // memory copy or reset
//    cuda_ret = cudaMemcpy(in_d, in_h, lefts[0]->get_ele_num() * sizeof(float),
//                          cudaMemcpyHostToDevice);
//    assert(cuda_ret == cudaSuccess);
//    cuda_ret = cudaMemcpy(kernel_d, kernel_h, weights.get_ele_num() * sizeof(float),
//                          cudaMemcpyHostToDevice);
//    assert(cuda_ret == cudaSuccess);
//    cuda_ret = cudaMemcpy(bias_d, bias, out_channels * sizeof(float),
//                          cudaMemcpyHostToDevice);
//    assert(cuda_ret == cudaSuccess);
//    cuda_ret = cudaMemset(out_d, 0, rights[0]->get_ele_num() * sizeof(float));
//    assert(cuda_ret == cudaSuccess);
//    cudaDeviceSynchronize();
//
//    for(int batch=0;batch<lefts[0]->batchSize;batch++){
//        conv_gpu_stride<<<dim_grid,dim_block>>>(
//                out_d+batch*(out_width*out_height*out_channels),
//                in_d+batch*(in_width*in_height*in_channels),
//                kernel_d,in_width,in_height,out_width,out_height,
//                kernel_size,in_channels,out_channels,w_stride,h_stride,bias_d);
//    }
//
//    cuda_ret = cudaDeviceSynchronize();
//    assert(cuda_ret == cudaSuccess);
//
//    cuda_ret = cudaMemcpy(rights[0]->_data, out_d, rights[0]->get_ele_num() * sizeof(float),
//                          cudaMemcpyDeviceToHost);
//    assert(cuda_ret == cudaSuccess);
//    cudaDeviceSynchronize();
//
//    cudaFree(in_d);
//    cudaFree(out_d);
//    cudaFree(kernel_d);
//    cudaFree(bias_d);
}

void ConvLayer::bp_gpu(std::vector<Blob *> lefts, std::vector<Blob *> rights) {
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
    manage_mem_and_conv(warped_right,warped_kernel,*lefts[0],1,1);
    update(lefts[0]->batchSize);
}

void ConvLayer::manage_mem_and_conv(Blob &input, Blob &kernel, Blob &output, int w_stride, int h_stride, float* bias){

    int batch_size = input.batchSize;
    int in_width = input.x;
    int in_height = input.y;
    int kernel_size = kernel.x;
    int in_channels = input.z;
    int out_channels = kernel.batchSize;
    int out_width = (in_width - kernel_size) / w_stride + 1;
    int out_height = (in_height - kernel_size) / h_stride + 1;

    if(bias==NULL){
        bias=new float[out_channels];
        memset(bias,0,out_channels*sizeof(float));
    }

    output.reset();
    cudaError_t cuda_ret;
    dim3 dim_grid,dim_block;
    dim_block.x=16; dim_block.y=16; dim_block.z=1;
    dim_grid.x=(out_width-1)/16+1; dim_grid.y=(out_width-1)/16+1; dim_grid.z=out_channels;

    float *in_h,*in_d,*kernel_h,*kernel_d,*out_d,*bias_d;

    // device memory alloc
    in_h=input._data;
    kernel_h=kernel._data;
    cuda_ret = cudaMalloc((void**)&in_d, input.get_ele_num() * sizeof(float));
    assert(cuda_ret == cudaSuccess);

    cuda_ret = cudaMalloc((void**)&kernel_d, kernel.get_ele_num() * sizeof(float));
    assert(cuda_ret == cudaSuccess);

    cuda_ret = cudaMalloc((void**)&out_d, output.get_ele_num() * sizeof(float));
    assert(cuda_ret == cudaSuccess);

    cuda_ret = cudaMalloc((void**)&bias_d, out_channels * sizeof(float));
    assert(cuda_ret == cudaSuccess);

    cudaDeviceSynchronize();

    // memory copy or reset
    cuda_ret = cudaMemcpy(in_d, in_h, input.get_ele_num() * sizeof(float),
                          cudaMemcpyHostToDevice);
    assert(cuda_ret == cudaSuccess);

    cuda_ret = cudaMemcpy(kernel_d, kernel_h, kernel.get_ele_num() * sizeof(float),
                          cudaMemcpyHostToDevice);
    assert(cuda_ret == cudaSuccess);

    cuda_ret = cudaMemcpy(bias_d, bias, out_channels * sizeof(float),
                          cudaMemcpyHostToDevice);
    assert(cuda_ret == cudaSuccess);
    cuda_ret = cudaMemset(out_d, 0, output.get_ele_num() * sizeof(float));
    assert(cuda_ret == cudaSuccess);
    cudaDeviceSynchronize();

    for(int batch=0;batch<input.batchSize;batch++){
        conv_gpu_stride<<<dim_grid,dim_block>>>(
                out_d+batch*(out_width*out_height*out_channels),
                        in_d+batch*(in_width*in_height*in_channels),
                        kernel_d,in_width,in_height,out_width,out_height,
                        kernel_size,in_channels,out_channels,w_stride,h_stride,bias_d);
    }

    cuda_ret = cudaDeviceSynchronize();
    assert(cuda_ret == cudaSuccess);

    cuda_ret = cudaMemcpy(output._data, out_d, output.get_ele_num() * sizeof(float),
                          cudaMemcpyDeviceToHost);
    assert(cuda_ret == cudaSuccess);
    cudaDeviceSynchronize();

    cudaFree(in_d);
    cudaFree(out_d);
    cudaFree(kernel_d);
    cudaFree(bias_d);
}