/***
 * @file generator.h
 * @author Quan Fan
 * @brief Generator
 * @date 08/Mar/2019
 */

#include "blob.h"

/***
 * @brief This is the base class of generators
 * 
 */
class Generator
{
    virtual Blob* next_batch()=0;
};