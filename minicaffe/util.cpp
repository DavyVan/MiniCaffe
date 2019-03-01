/***
 * @file util.cpp
 * @author Quan Fan
 * @brief Implementation of util.h
 * @date 01/Mar/2019
 */

#include<cstring>
#include "util.h"

void alloc_and_strcpy(char** dst, char* src)
{
    *dst = new char[strlen(src)];
    strcpy(*dst, src);
}