/***
 * @file blob.h
 * @author Quan Fan
 * @brief Defines the Blob class.
 * @date 20/Feb/2019
 */

#ifndef _BLOB_H_
#define _BLOB_H_

/***
 * @brief Blob class, where the data actually stored.
 * 
 */
class Blob
{
    public:
        char* name;         /**< Blob's name */
        int batchSize;      /**< Batch size that current blob contains. This is the 1st dimension of blob. */
        int x;              /**< x dimension. Blob describes a 3D matrix. This is the 2nd dimension of blob*/
        int y;              /**< y dimension. Blob describes a 3D matrix. This is the 3rd dimension of blob*/
        int z;              /**< z dimension. Blob describes a 3D matrix. This is the 4th dimension of blob */
        int sizeofEle;      /**< Size in bytes of each element. */
        
        float* _data;       /**< Where actually the data stored. We now only consider float type. */

        Blob(char* name, int sizeofEle=4);

        Blob(char* name, int batchSize, int x, int y, int z, int sizeofEle=4);
        // ~Blob();

        int get_ele_num();
};

#endif