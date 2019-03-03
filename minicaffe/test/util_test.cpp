/***
 * @file util_test.cpp
 * @author Quan Fan
 * @brief Test file of utilities functions.
 * @date 01/Mar/2019
 */

#include "gtest/gtest.h"
#include "../util.h"

TEST(UtilTest, alloc_and_strcpy)
{
    char* dst = NULL;
    char* src = "thisissrc";

    alloc_and_strcpy(&dst, src);

    ASSERT_NE(dst, nullptr);
    ASSERT_STREQ(dst, src);
}