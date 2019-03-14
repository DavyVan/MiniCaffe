/***
 * @file blob.cpp
 * @author Quan Fan
 * @brief Implementation of Blob class
 * @date 20/Feb/2019
 */

#include <cstring>
#include <cstdlib>
#include "blob.h"
#include "util.h"
#include "errors.h"

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
        return 0;     // TODO: may call reset() or throw an error

    // Allocate memory space
    _data = new float[get_ele_num()*sizeofEle];

    //TODO: Initial value?

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