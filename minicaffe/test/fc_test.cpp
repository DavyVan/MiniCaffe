/***
 * @file fc_text.cpp
 * @author Quan Fan
 * @brief 
 * @date 18/Mar/2019
 */

#include "gtest/gtest.h"
#include "../layers/fc_layer.h"

TEST(FCLayerTest, get_outputs_dimensions)
{
    int leftDim[4] = {32, 16, 16, 3};
    int rightDim[4];
    FCLayer fc = FCLayer("fc", 100, true);
    fc.get_outputs_dimensions(leftDim, 1, rightDim, 1);
    EXPECT_EQ(32, rightDim[0]);
    EXPECT_EQ(1, rightDim[1]);
    EXPECT_EQ(100, rightDim[2]);
    EXPECT_EQ(1, rightDim[3]);
}

TEST(FCLayerTest, infer)
{
    std::vector<Blob*> lefts;
    Blob* left = new Blob("left", 32, 16, 16, 3);
    lefts.push_back(left);
    std::vector<Blob*> rights;
    Blob* right = new Blob("right", 32, 1, 100, 1);
    rights.push_back(right);

    FCLayer fc = FCLayer("fc", 100, true);
    int leftDim[4] = {32, 16, 16, 3};
    int rightDim[4];
    fc.get_outputs_dimensions(leftDim, 1, rightDim, 1);
    fc.init();
    left->init();
    right->init();
    fc.infer_gpu(lefts, rights);
}

TEST(FCLayerTest, bp)
{
    std::vector<Blob*> lefts;
    Blob* left = new Blob("left", 32, 16, 16, 3);
    lefts.push_back(left);
    std::vector<Blob*> rights;
    Blob* right = new Blob("right", 32, 1, 100, 1);
    rights.push_back(right);

    FCLayer fc = FCLayer("fc", 100, true);
    int leftDim[4] = {32, 16, 16, 3};
    int rightDim[4];
    fc.get_outputs_dimensions(leftDim, 1, rightDim, 1);
    fc.init();
    left->init();
    right->init();
    fc.infer_gpu(lefts, rights);
    fc.bp_gpu(lefts, rights);
}
