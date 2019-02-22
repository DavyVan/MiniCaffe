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
        char* name;
        int x;
        int y;
        int z;
        int elementSize;

        Blob(); // this is a placeholder for test
        // ~Blob();

        int get_ele_num();
};

#endif