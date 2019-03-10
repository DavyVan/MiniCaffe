#include <cstdio>
#include "minicaffe/blob.h"
#include "minicaffe/layers/mnist_generator.h"
#include <assert.h>
int main()
{
    // TODO: for input layer, must assign "data" before "laber"
    Blob blob = Blob();
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


    MnistGenerator generator=MnistGenerator("../../train-images.idx3-ubyte","../../train-labels.idx1-ubyte");
    std::vector<Blob> sample=generator.loadSample(32);
    assert(sample[0].batchSize==32);
}