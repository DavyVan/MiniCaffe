/***
 * @file sigmoid_cross_entropy_loss_test.cpp
 * @author Quan Fan
 * @brief 
 * @date 20/Mar/2019
 */

#include "gtest/gtest.h"
#include "../layers/sigmoid_cross_entropy_loss.h"

TEST(SigmoidCrossEntropyLossTest, infer)
{
    Blob* left_data = new Blob("data", 500, 10, 1, 1);
    Blob* left_label = new Blob("label", 500, 10, 1, 1);
    Blob* loss = new Blob("loss");

    SigmoidCrossEntropyLoss losslayer("losslayer");
    int inDims[] = {500, 10, 1, 1, 500, 10, 1, 1};
    int outDims[4];
    losslayer.get_outputs_dimensions(inDims, 2, outDims, 1);
    losslayer.init();
    losslayer.infer(std::vector<Blob*>({left_data, left_label}), std::vector<Blob*>({loss}));
}

TEST(SigmoidCrossEntropyLossTest, bp)
{
    Blob* left_data = new Blob("data", 500, 10, 1, 1);
    Blob* left_label = new Blob("label", 500, 10, 1, 1);
    Blob* loss = new Blob("loss");

    SigmoidCrossEntropyLoss losslayer("losslayer");
    int inDims[] = {500, 10, 1, 1, 500, 10, 1, 1};
    int outDims[4];
    losslayer.get_outputs_dimensions(inDims, 2, outDims, 1);
    losslayer.init();
    losslayer.infer(std::vector<Blob*>({left_data, left_label}), std::vector<Blob*>({loss}));
    losslayer.bp(std::vector<Blob*>({left_data, left_label}), std::vector<Blob*>({loss}));
}