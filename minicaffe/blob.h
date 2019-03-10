/***
 * @file blob.h
 * @author Quan Fan
 * @brief Defines the Blob class.
 * @date 20/Feb/2019
 */

#ifndef _BLOB_H_
#define _BLOB_H_

#include <cstdlib>
// #include "layer.h"

class Layer;

/***
 * @brief Blob class, where the data actually stored.
 * 
 */
class Blob
{
    public:
        char* name;                 /**< Blob's name */
        int batchSize;              /**< Batch size that current blob contains. This is the 1st dimension of blob. */
        int x;                      /**< x dimension. Blob describes a 3D matrix. This is the 2nd dimension of blob*/
        int y;                      /**< y dimension. Blob describes a 3D matrix. This is the 3rd dimension of blob*/
        int z;                      /**< z dimension. Blob describes a 3D matrix. This is the 4th dimension of blob */
        int sizeofEle;              /**< Size in bytes of each element. */

        float* _data = NULL;        /**< Where actually the data stored. We now only consider float type. */

        Layer* left_layer = NULL;   /**< Connected layer on the left. One blob can only have one left layer. */
        Layer* right_layer = NULL;  /**< Connected layer on the right. One blob can only have one right layer. */

        /***
         * @brief Constructor. Call this when you don't know the size of blob
         * 
         * @param name      Name of blob.
         * @param sizeofEle sizeof element. Default is float(4)
         */
        Blob(const char* name, int sizeofEle=4);

        /***
         * @brief Constructor. Call this when you are aware of each dimensions of blob
         * 
         * @param name      Name of blob
         * @param batchSize 1st dimension, see @Blob::batchSize.
         * @param x         2nd dimension, see @Blob::x.
         * @param y         3rd dimension, see @Blob::y.
         * @param z         4th dimension, see @Blob::z.
         * @param sizeofEle Size of each element, see @Blob::sizeofEle.
         * 
         */
        Blob(const char* name, int batchSize, int x, int y, int z, int sizeofEle=4);

        /***
         * @brief No-param constructor.
         * This is a placeholder function and not suppose to be used in practice.
         */
        Blob();

        /***
         * @brief De-constructor.
         * Free all allocated memory.
         * 
         */
        ~Blob();

        int get_ele_num();

        /***
         * @brief Initialize current blob.
         * Allocate memory space and assign the first value.
         * 
         * @return Exit code
         */
        int init();
};

#endif