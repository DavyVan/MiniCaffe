#include <cstdio>
#include "minicaffe/minicaffe.h"
#include <assert.h>
int main()
{

    // 1. Init data generator with target file / folder / database etc.
    MnistGenerator generator("./train-images.idx3-ubyte", "./train-labels.idx1-ubyte");

    // 2. Create an empty net and initialize it with a generator
    Net net(&generator);

    // 3. Add some layers


    // 3. Init net
    net.init();

    // 4. train with CPU
    net.train();
    // or with GPU if any layer supports
    net.train(true);

    // 5. get result
    Blob* output = net.get_output("loss");
}