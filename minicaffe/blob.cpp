/***
 * @file blob.cpp
 * @author Quan Fan
 * @brief Implementation of Blob class
 * @date 20/Feb/2019
 */

#include <cstring>
#include "blob.h"
#include "util.h"

Blob::Blob(char* name, int sizeofEle)
    : sizeofEle(sizeofEle)
{
    alloc_and_strcpy(&(this->name), name);
}

Blob::Blob()
{
    x = 0;
    y = 0;
    z = 0;
    sizeofEle = 4;
    name = "undefined";
}

Blob::~Blob()
{
    if (_data != NULL)
        delete _data;
}

int Blob::get_ele_num()
{
    return x*y*z;
}