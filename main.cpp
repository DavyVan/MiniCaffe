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


    FCLayer fc1("fc1", 100, true);
    const char* fc1_lefts[] = {"data"};
    const char* fc1_rights[] = {"fc1"};
    net.add_layer(&fc1, fc1_lefts, 1, fc1_rights, 1);
    printf("fc1 layer added\n");

    FCLayer fc2("fc2", 1, true);
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
    net.train(10);
    // or with GPU if any layer supports
    // net.train(true);

    // 5. get result
    // Blob* output = net.get_output("loss");
}