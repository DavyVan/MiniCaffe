#include <cstdio>
#include "minicaffe/minicaffe.h"
#include <assert.h>
int main()
{

    // 1. Init data generator with target file / folder / database etc.
    MnistGenerator generator("../../train-images.idx3-ubyte", "../../train-labels.idx1-ubyte");
    printf("file loaded\n");

    // 2. Create an empty net and initialize it with a generator
    Net net(&generator, 2);
    printf("net created\n");

    // 3. Add some layers
    // Input layer
    InputLayer input("input");
    const char* input_lefts[] = {"sample_temp", "label_temp"};
    const char* input_rights[] = {"data", "label"};
    net.add_layer(&input, input_lefts, 0, input_rights, 2);
    printf("input layer added\n");

    // conv1
    ConvLayer conv1("conv1", 28, 28, 5, 1, 256, 3, 3);
    const char* conv1_lefts[] = {"data"};
    const char* conv1_rights[] = {"conv1"};
    net.add_layer(&conv1, conv1_lefts, 1, conv1_rights, 1);

    // pooling
    PoolingLayer pool1("pool1", 2, 2, 1);
    const char* pool1_lefts[] = {"conv1"};
    const char* pool1_rights[] = {"pool1"};
    net.add_layer(&pool1, pool1_lefts, 1, pool1_rights, 1);

    // relu1
    ReluLayer relu1("relu1");
    const char* relu1_lefts[] = {"pool1"};
    const char* relu1_rights[] = {"relu1"};
    net.add_layer(&relu1, relu1_lefts, 1, relu1_rights, 1);

    FCLayer fc1("fc1", 100, true);
    const char* fc1_lefts[] = {"relu1"};
    const char* fc1_rights[] = {"fc1"};
    net.add_layer(&fc1, fc1_lefts, 1, fc1_rights, 1);
    printf("fc1 layer added\n");

    FCLayer fc2("fc2", 10, true);
    const char* fc2_lefts[] = {"fc1"};
    const char* fc2_rights[] = {"fc2"};
    net.add_layer(&fc2, fc2_lefts, 1, fc2_rights, 1);
    printf("fc2 layer added\n");

    SigmoidCrossEntropyLoss losslayer("loss");
    const char* loss_lefts[] = {"fc2", "label"};
    const char* loss_rights[] = {"loss"};
    net.add_layer(&losslayer, loss_lefts, 2, loss_rights, 1);
    printf("loss layer added\n");

    // 3. Init net
    net.init();
    printf("inited\n");

    // 4. train with CPU
    net.train(20);
    // print_matrix(fc1.weight, 1, fc1.K_, fc1.N_, 1);
    // helper::print_blob(*net.blobs[net.get_blob_id_by_name("fc1")]);
    // or with GPU if any layer supports
    // net.train(true);

    // 5. get result
    // Blob* output = net.get_output("loss");
}