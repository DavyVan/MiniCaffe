/***
 * @file net.cpp
 * @author Quan Fan
 * @brief Implementation of Net class
 * @date 01/Mar/2019
 */

#include <cstring>
#include "net.h"
#include "errors.h"

int Net::add_layer(Layer* layer, const char* lefts[], const char* rights[])
{
    // Do some checks
    bool isOK = false;
    int numTail = tail.size();
    int numLefts = sizeof(lefts) / sizeof(char*);

    // Check lefts with tail
    for (int i = 0; i < numTail; i++)
    {
        for (int j = 0; j < numLefts; j++)
        {
            if (strcmp(tail[i]->name, lefts[j]) == true)
            {
                isOK = true;
                break;
            }
        }
    }
    // TODO: we now only consider tail, if not match, then abort.
    if (isOK == false)  // No match
    {
        print_err_str(LEFT_NOT_MATCH);
        return LEFT_NOT_MATCH;
    }
    isOK = false;

    // Check right with all exist TODO: we will be sure no name can show up twice manually.

    // Check passed. Create blobs
    // Decide blob dimensions.
    int inputDims[numTail * 4];
    for (int i = 0; i < numTail; i++)
    {
        inputDims[i*4 + 0] = tail[i]->batchSize;
        inputDims[i*4 + 1] = tail[i]->x;
        inputDims[i*4 + 2] = tail[i]->y;
        inputDims[i*4 + 3] = tail[i]->z;
    }
    
}