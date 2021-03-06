//
// Created by mingchen on 3/14/19.
//

#include <cstdio>
#include "minicaffe/blob.h"
#include "minicaffe/layers/mnist_generator.h"
#include <assert.h>
#include <string>
#include "minicaffe/layers/conv_layer.h"
#include "minicaffe/util.h"
#include "minicaffe/layers/relu_layer.h"
#include "minicaffe/minicaffe.h"

void print_image( Blob &blob);
int main()
{
    // TODO: for input layer, must assign "data" before "laber"
    Blob blob = Blob();
    blob.batchSize=1;
    blob.x = 2;
    blob.y = 3;
    blob.z = 4;
    printf("Hi MiniCaffe is running! %d\n", blob.get_ele_num());

    /**
     * 1. Init data generator with target file/folder/database etc.
     * Generator generator = new MINSTGenerator("./minst");
     *
     * 2. Add layers
     * Net net();
     * net.update_generator(generator);
     * Net net(generator);
     * FCLayer.fc(net, ...);
     * ...
     *
     * 3. Init net
     * net.init();
     *
     * 4. train
     * net.train()
     *
     * 5. get reault
     * Blob output = net.get_output("output blob name");
     */

////
//    MnistGenerator generator=MnistGenerator("../train-images.idx3-ubyte","../train-labels.idx1-ubyte");
//    std::vector<Blob> sample=generator.loadSample(3);
//    print_image(sample[0]);
//    helper::print_blob(sample[1]);

    int in_batch=2;
    int in_width=4;
    int in_height=4;
    int kernel_size=2;
    int in_channels=1;
    int out_channels=2;
    int w_stride=2;
    int h_stride=2;
//
    ConvLayer conv((char*)"conv1",in_width,in_height,kernel_size,in_channels,out_channels,w_stride,h_stride);
    int* in_dim=new int[4]{in_batch,in_width,in_height,in_channels};
    int* out_dim=new int[4];
    conv.get_outputs_dimensions(in_dim,1,out_dim,1);

    Blob in_blob("input",in_batch,in_width,in_height,in_channels);
    Blob out_blob("output",out_dim[0],out_dim[1],out_dim[2],out_dim[3]);
    in_blob.init();
    out_blob.init();

    int multi=-1;
    for(int i=0;i<in_blob.get_ele_num();i++){
        in_blob._data[i]=i*multi;
        multi*=-1;
    }
    for(int i=0;i<conv.weights.get_ele_num();i++){
        conv.weights._data[i]=i;
    }

    vector<Blob*> left;
    vector<Blob*> right;
    left.push_back(&in_blob);
    right.push_back(&out_blob);

    conv.infer_gpu(left,right);
    helper::print_blob(in_blob);
    helper::print_blob(conv.weights);
    helper::print_blob(out_blob);

    conv.infer(left,right);
    helper::print_blob(in_blob);

    conv.bp(left,right);
    helper::print_blob(in_blob);
    helper::print_blob(conv.weights);
    helper::print_blob(out_blob);
    conv.bp_gpu(left,right);
    helper::print_blob(in_blob);
    helper::print_blob(conv.weights);
    helper::print_blob(out_blob);


//    ReluLayer relu("relu1");
//
//    int* in_dim=new int[4]{in_batch,in_width,in_height,in_channels};
//    int* out_dim=new int[4];
//    relu.get_outputs_dimensions(in_dim,1,out_dim,1);
//
//    Blob in_blob("input",in_batch,in_width,in_height,in_channels);
//    Blob out_blob("output",out_dim[0],out_dim[1],out_dim[2],out_dim[3]);
//    in_blob.init();
//    out_blob.init();
//
//    int multi=-1;
//    for(int i=0;i<in_blob.get_ele_num();i++){
//        in_blob._data[i]=i*multi;
//        out_blob._data[i]=-i*multi;
//        multi*=-1;
//    }
//
//    helper::print_blob(in_blob);
//    helper::print_blob(out_blob);
//
//    vector<Blob*> left;
//    vector<Blob*> right;
//    left.push_back(&in_blob);
//    right.push_back(&out_blob);
//
//    relu.bp_gpu(left,right);
//    helper::print_blob(in_blob);
//    helper::print_blob(conv.weights);
//    helper::print_blob(out_blob);
//
//    FCLayer fclayer("fc", 100, true);
//    fclayer.M_ = 2;
//    int *in_dim = new int[4]{in_batch, in_width, in_height, in_channels};
//    int *out_dim = new int[4];
//    fclayer.get_outputs_dimensions(in_dim, 1, out_dim, 1);
//    Blob in_blob("input", in_batch, in_width, in_height, in_channels);
//    Blob out_blob("output", out_dim[0], out_dim[1], out_dim[2], out_dim[3]);
//    in_blob.init();
//    out_blob.init();
//
//    int multi=-1;
//    for(int i=0;i<in_blob.get_ele_num();i++){
//        in_blob._data[i]=i*multi;
//        out_blob._data[i]=-i*multi;
//        multi*=-1;
//    }
//
//    helper::print_blob(in_blob);
//    helper::print_blob(out_blob);
//
//    vector<Blob*> left;
//    vector<Blob*> right;
//    left.push_back(&in_blob);
//    right.push_back(&out_blob);
//
//    fclayer.infer(left, right);
//    printf("bbbb\n");
//    // helper::print_blob(*right[0]);
//    fclayer.bp(left, right);
//    // helper::print_blob(*left[0]);
//    relu.bp_gpu(left,right);
//    helper::print_blob(in_blob);
//    helper::print_blob(out_blob);

    // FCLayer fclayer("fc", 100, true);
    // fclayer.M_ = 2;
    // int *in_dim = new int[4]{in_batch, in_width, in_height, in_channels};
    // int *out_dim = new int[4];
    // fclayer.get_outputs_dimensions(in_dim, 1, out_dim, 1);
    // Blob in_blob("input", in_batch, in_width, in_height, in_channels);
    // Blob out_blob("output", out_dim[0], out_dim[1], out_dim[2], out_dim[3]);
    // in_blob.init();
    // out_blob.init();
    // fclayer.init();

    // printf("blobs inited\n");
    // int multi = -1;
    // for (int i = 0; i < in_blob.get_ele_num(); i++)
    // {
    //     in_blob._data[i] = i * multi / 100.0;
    //     multi *= -1;
    // }
    // // for (int i = 0; i < fclayer.K_*fclayer.N_; i++)
    // // {
    // //     fclayer.weight[i] = i / 100.0;
    // // }

    // vector<Blob *> left;
    // vector<Blob *> right;
    // left.push_back(&in_blob);
    // right.push_back(&out_blob);
    // printf("aaaaaa\n");

    // fclayer.infer(left, right);
    // printf("bbbb\n");
    // // helper::print_blob(*right[0]);
    // fclayer.bp(left, right);
    // // helper::print_blob(*left[0]);
}

#include <iostream>
//void print_blob( Blob &blob){
//    using namespace std;
//    cout<<"print blob "<<blob.name<<endl;
//    for(int batch=0;batch<blob.batchSize;batch++){
//        cout<<"batch "<<batch<<endl;
//        for(int channel=0;channel<blob.z;channel++){
//            cout<<"channel "<<channel<<endl;
//            for(int y=0;y<blob.y;y++){
//                for(int x=0;x<blob.x;x++){
//                    cout<<blob(batch,x,y,channel)<<" ";
//                }
//                cout<<endl;
//            }
//
//        }
//    }
//    cout<<endl;
//}
void print_image( Blob &blob){
    using namespace std;
    cout<<"print blob "<<blob.name<<endl;
    for(int batch=0;batch<blob.batchSize;batch++){
        cout<<"batch "<<batch<<endl;
        for(int channel=0;channel<blob.z;channel++){
            cout<<"channel "<<channel<<endl;
            for(int y=0;y<blob.y;y++){
                for(int x=0;x<blob.x;x++){
                    if(!helper::float_eq(blob(batch,x,y,channel),0))
                        cout<<".";
                    else
                        cout<<"o";
                }
                cout<<endl;
            }

        }
    }
    cout<<endl;
}

