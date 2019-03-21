//
// Created by mingchen on 3/9/19.
//

#ifndef MAIN_MNIST_GENERATOR_H
#define MAIN_MNIST_GENERATOR_H

#include "../blob.h"
#include <string>
#include <fstream>
#include <vector>
class MnistGenerator {
public:

    virtual std::vector<Blob> loadSample(int batchSize);
    int reset();
    MnistGenerator(std::string sampleFile,std::string labelFile);

private:
    int offset;
    int size;
    int row_size,col_size;
    std::ifstream sampleFS;
    std::ifstream labelFS;
    void loadToMemory();
    std::vector<std::vector<float> > _images;
    std::vector<int> _labels;
    inline int to_int(char* buffer){
        return ((buffer[0] & 0xff) << 24) | ((buffer[1] & 0xff) << 16) |
               ((buffer[2] & 0xff) << 8) | ((buffer[3] & 0xff) << 0);
    }
};


#endif //MAIN_MNIST_GENERATOR_H
