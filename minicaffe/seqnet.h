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
#include "generator.h"

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
        SeqNet();

        /***
         * @brief Instantiate with generator.
         * 
         * @param generator The generator.
         */
        SeqNet(Generator* generator);

        /***
         * @brief Change the data generator.
         * 
         * @param generator The generator.
         */
        void update_generator(Generator* generator);

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
         *        Initialization includes allocation of memory, and reset all the blobs or variables to proper values.
         * 
         * @return The error code. 0 for normal.
         */
        int init();

        void infer();

        void bp();

        /***
         * @brief Training process consists of inference and BP.
         * 
         */
        void train();

        Blob get_output(char* name);

    private:
        vector<Layer*> layers;          /**< All layers, in the order of insertion and running */
        vector<Blob*> blobs;            /**< All blobs, in the order of insertion */
        vector<vector<Blob*>> lefts;    /**< lefts for each layer */
        vector<vector<Blob*>> rights;   /**< rights for each layer */
        Generator* dataGenerator;

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
#endif
};
typedef SeqNet Net;

#endif