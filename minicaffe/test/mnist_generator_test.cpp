//
// Created by mingchen on 3/10/19.
//

#include "gtest/gtest.h"
#include "../layers/mnist_generator.h"

TEST(MnistGeneratorTest,file_exist)
{
    MnistGenerator generator=MnistGenerator("../../../train-images.idx3-ubyte","../../../train-labels.idx1-ubyte");
    std::vector<Blob> sample=generator.loadSample(32);
    ASSERT_EQ(sample[0].batchSize, 32);

}
