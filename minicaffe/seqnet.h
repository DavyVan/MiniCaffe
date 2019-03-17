/***
 * @file seqnet.h
 * @author Quan Fan
 * @brief New Net
 * @date 08/Mar/2019
 */
#ifdef _NET_H_
#error net.h is obsoleted. Do not use it anywhere.
#endif

#ifndef _SEQNET_H_
#define _SEQNET_H_

#include <vector>
#include "blob.h"
#include "layers/mnist_generator.h"

#ifdef TEST_ENABLED
#include "gtest/gtest_prod.h"
#endif

using std::vector;

class Layer;

class SeqNet
{
    public:

        /***
         * @brief Non-arg constructor. Keep it for test.
         * 
         */
        SeqNet(int batchsize=32);

        /***
         * @brief Instantiate with generator.
         * 
         * @param generator The generator.
         */
        SeqNet(MnistGenerator* generator, int batchsize=32);

        /***
         * @brief Change the data generator.
         * 
         * @param generator The generator.
         */
        void update_generator(MnistGenerator* generator);

        /***
         * @brief Add new layer to this net.
         * This function will create necessary blobs to connect layers. It will firstly check the tail pointer by lefts' name and then search the net if not match.
         * If any designated lefts do not exist, function will report and abort. If any right blob has the same name as existed blob, report and abort.
         * NOTICE: Some limitations apply, please see README
         * 
         * @param layer     The layer to be added to this net.
         * @param lefts     The name set of left blobs.
         * @param numLefts  The number of left blobs.
         * @param rights    The name set of right blobs.
         * @param numRights The number of right blobs.
         * 
         * @return The error code.
         */
        int add_layer(Layer* layer, const char* lefts[], const int numLefts, const char* rights[], const int numRights);

        /***
         * @brief Initialize neural network before running.
         *        Call init() of each blob and layer.
         * 
         * @return The error code. 0 for normal.
         */
        int init();

        /***
         * @brief Do forward.
         *        1. Reset all blobs
         *        2. Generate a new batch of samples
         *        3. Iterate each layer and call infer()
         *          (a) find lefts and rights blobs
         *          (b) call infer() 
         */
        void infer();

        /***
         * @brief Do backward propagation
         * 
         */
        void bp();

        /***
         * @brief Training process consists of inference and BP.
         *        Call infer() and bp()
         * 
         */
        void train();

        /***
         * @brief return a copy of the required blob.
         * 
         */
        Blob* get_output(const char* name);

        /***
         * @brief Return the batchsize
         * 
         * @return int The batchsize
         */
        int static get_batchsize();

    private:
        vector<Layer*> layers;          /**< All layers, in the order of insertion and running */
        vector<Blob*> blobs;            /**< All blobs, in the order of insertion */
        vector<vector<Blob*>> lefts;    /**< lefts for each layer */
        vector<vector<Blob*>> rights;   /**< rights for each layer */
        MnistGenerator* dataGenerator;
        static int batchsize;

        /***
         * @brief As its name.
         * 
         * @param name      Name of target blob.
         * 
         * @return int      The index in @Net::blobs.
         */
        int get_blob_id_by_name(const char* name);

#ifdef TEST_ENABLED
        FRIEND_TEST(SeqNetTest, add_layer_1layer_0in2out);
        FRIEND_TEST(SeqNetTest, add_layer_3layers);
        FRIEND_TEST(SeqNetTest, get_blob_id_by_name);
        FRIEND_TEST(SeqNetTest, infer);
        FRIEND_TEST(SeqNetTest, bp);
#endif
};
typedef SeqNet Net;

#endif