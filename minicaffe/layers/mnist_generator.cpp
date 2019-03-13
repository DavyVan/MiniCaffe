//
// Created by mingchen on 3/9/19.
//

#include "mnist_generator.h"
#include "../blob.h"
#include <iostream>
#include <assert.h>
#include <cstring>
#include <fstream>

MnistGenerator::MnistGenerator(std::string sampleFile, std::string labelFile) {
    using namespace std;
    try{
        sampleFS=ifstream(sampleFile.c_str(),ios::in|ios::binary);
        labelFS=ifstream(labelFile.c_str(),ios::in|ios::binary);
    }
    catch(exception e){
        std::cout<<"[ERROR] Sample file not exist"<<endl<<flush;
        std::exit(0);
    }
    char buffer[4];

    sampleFS.read(buffer,4);
    int magic_number_sample=to_int(buffer);
    assert(magic_number_sample==0x803);

    labelFS.read(buffer,4);
    int magic_number_label=to_int(buffer);
    assert(magic_number_label==0x801);

    sampleFS.read(buffer,4);
    int sample_size=to_int(buffer);
    labelFS.read(buffer,4);
    int label_size=to_int(buffer);
    assert(sample_size==label_size);
    size=sample_size;

    sampleFS.read(buffer,4);
    row_size=to_int(buffer);
    sampleFS.read(buffer,4);
    col_size=to_int(buffer);
    offset=0;

    try{
        loadToMemory();
    }
    catch (exception e){
        cout<<"[ERROR] Can't load file to memory!!"<<endl<<flush;
        std::exit(0);
    }


}

std::vector<Blob> MnistGenerator::loadSample(int batchSize){
    int valid_batch=std::min(size-offset,batchSize);
    Blob sample("sample_temp",valid_batch,row_size,col_size,1,4);
    sample.init();
    Blob label("label_temp",valid_batch,1,1,1,4);
    label.init();
    for(int i=0;i<valid_batch;i++){
        std::cout<<i*row_size*col_size<<std::endl;
        std::cout<<&sample._data[i*row_size*col_size]<<std::endl;
        std::cout<<_images[offset+i].data()<<std::endl;

        memcpy(&sample._data[i*row_size*col_size],_images[offset+i].data(),row_size*col_size);
        //std::copy(_images[offset+i].begin(),_images[offset+i].end(),&sample._data[i*row_size*col_size]);
        label._data[i]=_labels[offset+i];
        std::cout<<i<<" label: "<<_labels[offset+i]<<std::endl;
    }
    std::vector<Blob> ret;
    ret.push_back(sample);
    ret.push_back(label);
    offset=offset+batchSize;
    if(offset>=size){
        reset();
    }
    return ret;
}

int MnistGenerator::reset() {
    offset=0;
    return 0;
}
void MnistGenerator::loadToMemory() {
    char* buffer=new char[row_size*col_size];
    for(int i=0;i<size;i++){
        sampleFS.read(buffer,row_size*col_size);
        std::vector<float> image(row_size*col_size);
        for(int j=0;j<row_size*col_size;j++){
            image[j]=float(buffer[j]/255.0);
        }
        _images.push_back(image);
    }
    delete[] buffer;

    buffer=new char[size];
    labelFS.read(buffer,size);
    for(int i=0;i<size;i++){
        _labels.push_back(float(buffer[i]));
    }

    delete[] buffer;
}