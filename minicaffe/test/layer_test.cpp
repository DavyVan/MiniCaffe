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
    int init(){return 0;}
    void infer(){}
    void bp(){}
    void get_outputs_dimensions(int inputs_dims[], const int numInputs, int outputs_dims[], const int numOutputs){}
    bool check_dimensions(){return true;}
};

TEST(LayerTest, constructor)
{
    testLayer t("ttttt");
    ASSERT_STREQ(t.name, "ttttt");
}