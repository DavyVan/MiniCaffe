#include <cstdio>
#include "minicaffe/blob.h"
#include "minicaffe/layers/mnist_generator.h"
#include <assert.h>
int main()
{
    // TODO: for input layer, must assign "data" before "laber"
    MnistGenerator generator=MnistGenerator("../../train-images.idx3-ubyte","../../train-labels.idx1-ubyte");
    std::vector<Blob> sample=generator.loadSample(32);
    assert(sample[0].batchSize==32);
}