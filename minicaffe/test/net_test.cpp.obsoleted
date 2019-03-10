#include "gtest/gtest.h"
#include "../net.h"
#include "../layer.h"
#include "../errors.h"

class TestLayerForNet : public Layer
{
public:
    int init(){return 0;}
    void infer(){}
    void bp(){}
    void get_outputs_dimensions(int inputs_dims[], const int numInputs, int outputs_dims[], const int numOutputs)
    {
        int size = numOutputs * 4;
        for (int i = 0; i < size; i++)
            outputs_dims[i] = i+1;
    }
    TestLayerForNet(char* name):Layer(name){}
    bool check_dimensions(){return true;}
};

TEST(NetTest, add_layer_1layer_0in2out)
{
    TestLayerForNet tLayer = TestLayerForNet("testlayerfornet");
    Net net = Net();

    const char* rights[] = {"right1", "right2"};
    // const char* lefts[] = {};
    net.add_layer(&tLayer, NULL, 0, rights, 2);

    /**
     * Check for layer:
     * left_blobs: size==0;
     * right_blobs: size==2;
     */
    EXPECT_EQ(0, tLayer.left_blobs.size());
    ASSERT_EQ(2, tLayer.right_blobs.size());

    /**
     * Check for blob1:
     * name==right1
     * batchSize==1
     * x==2
     * y==3
     * z==4
     * sizeofEle==4
     * left_layer==tLayer
     */
    Blob* blob1 = tLayer.right_blobs[0];
    EXPECT_STREQ("right1", blob1->name);
    EXPECT_EQ(1, blob1->batchSize);
    EXPECT_EQ(2, blob1->x);
    EXPECT_EQ(3, blob1->y);
    EXPECT_EQ(4, blob1->z);
    EXPECT_EQ(4, blob1->sizeofEle);
    EXPECT_EQ(&tLayer, blob1->left_layer);
}

/**
 *        ____(l1out2)_____________________
 *       /                                 \
 * layer1                                   Layer3---(l3out)
 *       \____(l1out1)___layer2___(l2out)__/
 */
TEST(NetTest, add_layer_3layers)
{
    TestLayerForNet tLayer1 = TestLayerForNet("testlayerfornet1");
    TestLayerForNet tLayer2 = TestLayerForNet("testlayerfornet2");
    TestLayerForNet tLayer3 = TestLayerForNet("testlayerfornet3");
    Net net = Net();

    // add 1st layer
    const char* rights1[] = {"l1out1", "l1out2"};
    net.add_layer(&tLayer1, NULL, 0, rights1, 2);

    // add 2nd layer
    const char* lefts2[] = {"l1out1"};
    const char* rights2[] = {"l2out"};
    net.add_layer(&tLayer2, lefts2, 1, rights2, 1);

    // add 3rd layer
    const char* lefts3[] = {"l1out2", "l2out"};
    const char* rights3[] = {"l3out"};
    net.add_layer(&tLayer3, lefts3, 2, rights3, 1);

    /**
     * Check for layers' in/out blobs number
     */
    ASSERT_EQ(0, tLayer1.left_blobs.size());
    ASSERT_EQ(2, tLayer1.right_blobs.size());
    ASSERT_EQ(1, tLayer2.left_blobs.size());
    ASSERT_EQ(1, tLayer2.right_blobs.size());
    ASSERT_EQ(2, tLayer3.left_blobs.size());
    ASSERT_EQ(1, tLayer3.right_blobs.size());

    /**
     * Check for blob "l1out1"
     */
    ASSERT_EQ(tLayer1.right_blobs[0], tLayer2.left_blobs[0]);
    Blob* l1out1 = tLayer1.right_blobs[0];
    EXPECT_EQ(&tLayer1, l1out1->left_layer);
    EXPECT_EQ(&tLayer2, l1out1->right_layer);
    /**
     * l1out2
     */
    ASSERT_EQ(tLayer1.right_blobs[1], tLayer3.left_blobs[0]);
    Blob* l1out2 = tLayer1.right_blobs[1];
    EXPECT_EQ(&tLayer1, l1out2->left_layer);
    EXPECT_EQ(&tLayer3, l1out2->right_layer);
    /**
     * l2out
     */
    ASSERT_EQ(tLayer2.right_blobs[0], tLayer3.left_blobs[1]);
    Blob* l2out = tLayer2.right_blobs[0];
    EXPECT_EQ(&tLayer2, l2out->left_layer);
    EXPECT_EQ(&tLayer3, l2out->right_layer);
    /**
     * l3out
     */
    Blob* l3out = tLayer3.right_blobs[0];
    EXPECT_EQ(&tLayer3, l3out->left_layer);
    EXPECT_EQ(nullptr, l3out->right_layer);
}