/***
 * @file blob.cpp
 * @author Quan Fan
 * @brief Implementation of Blob class
 * @date 20/Feb/2019
 */

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "blob.h"
#include "util.h"
#include "errors.h"
#include <stdio.h>
Blob::Blob(const char* name, int sizeofEle)
    : sizeofEle(sizeofEle)
{
    alloc_and_strcpy(&(this->name), name);
}

Blob::Blob(const char* name, int batchSize, int x, int y, int z, int sizeofEle)
    : batchSize(batchSize), x(x), y(y), z(z), sizeofEle(sizeofEle)
{
    alloc_and_strcpy(&(this->name), name);
}

Blob::Blob()
{
    x = 0;
    y = 0;
    z = 0;
    sizeofEle = 4;
    alloc_and_strcpy(&(this->name), (char*)"undefined");
}

Blob::~Blob()
{
    if (this->_data != NULL)
        delete[] _data;
}

int Blob::get_ele_num()
{
    return batchSize*x*y*z;
}

int Blob::init()
{
    // check dimensions are non-zero
    if (batchSize == 0 || x==0 || y==0 || z==0)
    {
        print_err_str(ZERO_DIM);
        return ZERO_DIM;
    }

    // if blob has been already initialized
    if (_data != NULL)
    {
        reset();
        return 0; 
    }

    // Allocate memory space
    _data = new float[get_ele_num()*sizeofEle];
    reset();

    return 0;
}

Blob::Blob(const Blob &rhs)
        : batchSize(rhs.batchSize), x(rhs.x), y(rhs.y), z(rhs.z), sizeofEle(rhs.sizeofEle){
    alloc_and_strcpy(&(this->name), rhs.name);
    if(rhs._data!=NULL){
        if(this->_data!=NULL)
            delete[] this->_data;
        this->init();
        memcpy(this->_data,rhs._data,this->get_ele_num()*sizeofEle);
    }
}

Blob Blob::operator=(const Blob & rhs) {
    if(this->name!=NULL)
        delete this->name;
    alloc_and_strcpy(&(this->name), rhs.name);
    this->batchSize=rhs.batchSize;
    this->x=rhs.x;
    this->y=rhs.y;
    this->z=rhs.z;
    this->sizeofEle=rhs.sizeofEle;
    if(rhs._data!=NULL){
        if(this->_data!=NULL)
            delete[] this->_data;
        this->init();
        memcpy(this->_data,rhs._data,this->get_ele_num()*sizeofEle);
    }
    return *this;
}

float& Blob::operator()(int batch_pos, int x_pos, int y_pos, int z_pos) {
    if(_data==NULL){
        printf("[ERROR] Null Pointer when load blob %s (%d,%d,%d,%d)\n",name,batch_pos,x_pos,y_pos,z_pos);
        std::exit(-1);
    }
    if(batch_pos>=batchSize||x_pos>=x||y_pos>=y||z_pos>=z){
        printf("[ERROR] Index (%d,%d,%d,%d) out of range (%d,%d,%d,%d) in blob %s \n",batch_pos,x_pos,y_pos,z_pos,batchSize,x,y,z,name);
        std::exit(-1);
    }
    return _data[batch_pos*(x*y*z)+x_pos*(y*z)+y_pos*(z)+z_pos];
}