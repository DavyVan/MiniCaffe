/***
 * @file blob.cpp
 * @author Quan Fan
 * @brief Implementation of Blob class
 * @date 20/Feb/2019
 */

#include "blob.h"

Blob::Blob()
{
    x = 0;
    y = 0;
    z = 0;
    elementSize = 4;
    name = "undefined";
}

int Blob::get_ele_num()
{
    return x*y*z;
}