/***
 * @file layer_test.cpp
 * @author Quan Fan
 * @brief Test file of Layer class
 * @date 01/Mar/2019
 */

#include "gtest/gtest.h"
#include "../layer.h"

class testLayer : public Layer
{
public:
    testLayer(char* name):Layer(name){}
    int init(){}
    void infer(){}
    void bp(){}
    void get_outputs_dimension(int inputs_dims[], int outputs_dims[]){}
};

TEST(LayerTest, constructor)
{
    testLayer t("ttttt");
    ASSERT_STREQ(t.name, "ttttt");
}