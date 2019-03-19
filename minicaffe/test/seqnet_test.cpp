/***
 * @file seqnet_test.cpp
 * @author Quan Fan
 * @brief Test file of SeqNet
 * @date 08/Mar/2019
 */

#include "gtest/gtest.h"
#include "../seqnet.h"
#include "../layer.h"
#include "../errors.h"
#include "../layers/mnist_generator.h"

class TestLayerForSeqNet : public Layer
{
public:
    int init(){return 0;}
    void infer(vector<Blob*> lefts, vector<Blob*> rights){
        int numL = lefts.size();
        int numR = rights.size();

        if (numL < numR)
        {
            // copy left+1 to rights
            Blob* l = lefts[0];
            int elems = l->get_ele_num();
            for (int i = 0; i < numR; i++)
            {
                for (int di = 0; di < elems; di++)
                    rights[i]->_data[di] = l->_data[di] + 1;
            }
        }
        else if (numL == numR)
        {
            // copy left+2 to rights accordingly
            for (int i = 0; i < numL; i++)
            {
                int elems = lefts[0]->get_ele_num();
                for (int di = 0; di < elems; di++)
                    rights[i]->_data[di] = lefts[i]->_data[di] + 2;
            }
        }
        else if (numL > numR)
        {
            // add lefts then +1 to right
            int elems = lefts[1]->get_ele_num();
            Blob* r = rights[0];
            for (int di = 0; di < elems; di++)
            {
                float sumL = 0;
                for (int i = 0; i < numL; i++)
                    sumL += lefts[i]->_data[di];
                r->_data[di] = sumL + 1;
            }
        }
    }
    void bp(vector<Blob*> lefts, vector<Blob*> rights){
        vector<Blob*> t = lefts;
        lefts = rights;
        rights = t;

        int numL = lefts.size();
        int numR = rights.size();

        if (numL < numR)
        {
            // copy left+1 to rights
            Blob* l = lefts[0];
            int elems = 24;
            for (int i = 0; i < numR; i++)
            {
                for (int di = 0; di < elems; di++)
                    rights[i]->_data[di] = l->_data[di] + 1;
            }
        }
        else if (numL == numR)
        {
            // copy left+2 to rights accordingly
            for (int i = 0; i < numL; i++)
            {
                int elems = 24;
                for (int di = 0; di < elems; di++)
                    rights[i]->_data[di] = lefts[i]->_data[di] + 2;
            }
        }
        else if (numL > numR)
        {
            // add lefts then +1 to right
            int elems = 24;
            Blob* r = rights[0];
            for (int di = 0; di < elems; di++)
            {
                float sumL = 0;
                for (int i = 0; i < numL; i++)
                    sumL += lefts[i]->_data[di];
                r->_data[di] = sumL + 1;
            }
        }
    }
    void get_outputs_dimensions(int inputs_dims[], const int numInputs, int outputs_dims[], const int numOutputs)
    {
        int size = numOutputs * 4;
        for (int i = 0; i < size; i++)
            outputs_dims[i] = i+1;
    }
    TestLayerForSeqNet(char* name):Layer(name){}
    bool check_dimensions(){return true;}
};

class _Test_Generator : public MnistGenerator
{
    public:
    _Test_Generator():MnistGenerator("../../../train-images.idx3-ubyte","../../../train-labels.idx1-ubyte"){}
    std::vector<Blob> loadSample(int batchSize)
    {
        Blob *generated1 = new Blob("g1", 1, 2, 3, 4);
        Blob *generated2 = new Blob("g2", 1, 2, 3, 4);
        generated1->init();
        generated2->init();
        vector<Blob> ret;
        ret.push_back(*generated1);
        ret.push_back(*generated2);
        return ret;
    }
};

TEST(SeqNetTest, add_layer_1layer_0in2out)
{
    TestLayerForSeqNet tLayer = TestLayerForSeqNet("TestLayerForSeqNet");
    SeqNet net = SeqNet();

    const char* rights[] = {"right1", "right2"};
    net.add_layer(&tLayer, NULL, 0, rights, 2);

    /**
     * Check layers
     */
    EXPECT_EQ(1, net.layers.size());

    /**
     * Check blobs
     */
    EXPECT_EQ(2, net.blobs.size());

    /**
     * Check lefts
     */
    ASSERT_EQ(1, net.lefts.size());
    EXPECT_EQ(0, net.lefts[0].size());

    /**
     * Check rights
     */
    ASSERT_EQ(1, net.rights.size());
    EXPECT_EQ(2, net.rights[0].size());
    EXPECT_STREQ("right1", net.rights[0][0]->name);
    EXPECT_STREQ("right2", net.rights[0][1]->name);

    /**
     * Check the first blob
     */
    Blob* blob1 = net.rights[0][0];
    EXPECT_STREQ("right1", blob1->name);
    EXPECT_EQ(1, blob1->batchSize);
    EXPECT_EQ(2, blob1->x);
    EXPECT_EQ(3, blob1->y);
    EXPECT_EQ(4, blob1->z);
    EXPECT_EQ(4, blob1->sizeofEle);
}

/**
 *        ____(l1out2)_____________________
 *       /                                 \
 * layer1                                   Layer3---(l3out)
 *       \____(l1out1)___layer2___(l2out)__/
 */
TEST(SeqNetTest, add_layer_3layers)
{
    TestLayerForSeqNet tLayer1 = TestLayerForSeqNet("TestLayerForSeqNet1");
    TestLayerForSeqNet tLayer2 = TestLayerForSeqNet("TestLayerForSeqNet2");
    TestLayerForSeqNet tLayer3 = TestLayerForSeqNet("TestLayerForSeqNet3");
    SeqNet net = SeqNet();

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
     * Check layers
     */
    // total number
    ASSERT_EQ(3, net.layers.size());
    // layers' name
    EXPECT_STREQ("TestLayerForSeqNet1", net.layers[0]->name);
    EXPECT_STREQ("TestLayerForSeqNet2", net.layers[1]->name);
    EXPECT_STREQ("TestLayerForSeqNet3", net.layers[2]->name);

    /**
     * Check blobs
     */
    // size
    ASSERT_EQ(4, net.blobs.size());

    /**
     * Check lefts
     */
    // size
    ASSERT_EQ(3, net.lefts.size());
    EXPECT_EQ(0, net.lefts[0].size());
    EXPECT_EQ(1, net.lefts[1].size());
    EXPECT_EQ(2, net.lefts[2].size());
    // names of 3rd layer's lefts
    EXPECT_STREQ("l1out2", net.lefts[2][0]->name);
    EXPECT_STREQ("l2out", net.lefts[2][1]->name);

    /**
     * Check rights
     */
    // size
    ASSERT_EQ(3, net.rights.size());
    EXPECT_EQ(2, net.rights[0].size());
    EXPECT_EQ(1, net.rights[1].size());
    EXPECT_EQ(1, net.rights[2].size());
    // names of 1st layer's rights
    EXPECT_STREQ("l1out1", net.rights[0][0]->name);
    EXPECT_STREQ("l1out2", net.rights[0][1]->name);

    /**
     * Check blob's order, pick one of them
     */
    // pick one from blobs
    Blob* b = net.blobs[2];    // l2out
    EXPECT_EQ(b, net.rights[1][0]);
    EXPECT_EQ(b, net.lefts[2][1]);
}

TEST(SeqNetTest, get_blob_id_by_name)
{
    SeqNet net = SeqNet();
    Blob b1 = Blob("b1");
    Blob b2 = Blob("b2");
    Blob b3 = Blob("b3");
    Blob b4 = Blob("b4");

    net.blobs.push_back(&b1);
    net.blobs.push_back(&b2);
    net.blobs.push_back(&b3);
    net.blobs.push_back(&b4);

    EXPECT_EQ(0, net.get_blob_id_by_name("b1"));
    EXPECT_EQ(1, net.get_blob_id_by_name("b2"));
    EXPECT_EQ(2, net.get_blob_id_by_name("b3"));
    EXPECT_EQ(3, net.get_blob_id_by_name("b4"));
    EXPECT_EQ(-1, net.get_blob_id_by_name("b5"));
}

/**
 * No test for SeqNet::init(). init() of Blob or Layer should be tested in their own test suite.
 */

/**
 *                     ____(l1out2)_____________________
 *                    /                                 \
 * Generator == intput                                   Layer3---(l3out)
 *                    \____(l1out1)___layer2___(l2out)__/
 */
TEST(SeqNetTest, infer)
{
    TestLayerForSeqNet tLayer1 = TestLayerForSeqNet("TestLayerForSeqNet1");
    TestLayerForSeqNet tLayer2 = TestLayerForSeqNet("TestLayerForSeqNet2");
    TestLayerForSeqNet tLayer3 = TestLayerForSeqNet("TestLayerForSeqNet3");
    SeqNet net = SeqNet();

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

    // add generated blobs to lefts[0]
    _Test_Generator generator;
    net.update_generator(&generator);

    net.init();
    net.infer();

    // check "l3out"
    Blob* out = net.blobs[net.get_blob_id_by_name("l3out")];
    EXPECT_FLOAT_EQ(7, out->_data[0]);
}

TEST(SeqNetTest, bp)
{
    TestLayerForSeqNet tLayer1 = TestLayerForSeqNet("TestLayerForSeqNet1");
    TestLayerForSeqNet tLayer2 = TestLayerForSeqNet("TestLayerForSeqNet2");
    TestLayerForSeqNet tLayer3 = TestLayerForSeqNet("TestLayerForSeqNet3");
    SeqNet net = SeqNet();

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

    // add generated blobs to lefts[0]
    _Test_Generator generator;
    net.update_generator(&generator);

    net.init();
    net.infer();

    // check "l3out"
    Blob* out = net.blobs[net.get_blob_id_by_name("l3out")];
    ASSERT_FLOAT_EQ(7, out->_data[0]);

    net.bp();
    
    // Check "l1out1"
    out = net.blobs[net.get_blob_id_by_name("l1out1")];
    EXPECT_FLOAT_EQ(10, out->_data[0]);
}