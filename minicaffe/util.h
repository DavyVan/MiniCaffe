/***
 * @file util.h
 * @author Quan Fan
 * @brief Utilities functions.
 * @date 01/Mar/2019
 */
#include <cmath>
void alloc_and_strcpy(char**dst, const char* src);

namespace helper{
    inline bool float_eq(float a, float b){
        if (std::abs(a-b)<0.0001) return 1;
        return 0;
    }
}