/***
 * @file errors.cpp
 * @author Quan Fan
 * @brief See @errors.h
 * @date 01/Mar/2019
 */

#include <cstdio>
#include "errors.h"

char* err_str[] = {
    "Succeed.",                                          // 0
    "Blobs with any dimension that equals to zero",      // 1
    "Expected left blobs don't match any existed blob."  // 2
};

char* get_err_str(int _errno)
{
    return err_str[_errno];
}

void print_err_str(int _errno)
{
    printf("Error: %d - %s\n", _errno, get_err_str(_errno));
}