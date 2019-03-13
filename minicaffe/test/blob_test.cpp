#include "gtest/gtest.h"
#include "../blob.h"
#include "../errors.h"

TEST(BlobTest, get_ele_num_zero)
{
    Blob blob = Blob();
    blob.x = blob.y = blob.z = 0;

    ASSERT_EQ(0, blob.get_ele_num());
}
/*
TEST(BlobTest, get_ele_num_nonzero)
{
    Blob blob = Blob();
    blob.x = 1;
    blob.y = 2;
    blob.z = 3;

    ASSERT_EQ(6, blob.get_ele_num());
}
*/
TEST(BlobTest, constructor_1param)
{
    Blob b = Blob("blobname");
    EXPECT_STREQ("blobname", b.name);
    EXPECT_EQ(4, b.sizeofEle);
}

TEST(BlobTest, constructor_2params)
{
    Blob b = Blob("blobname", 3);
    EXPECT_STREQ("blobname", b.name);
    EXPECT_EQ(3, b.sizeofEle);
}

TEST(BlobTest, constructor_fullparams)
{
    Blob b = Blob("blobname", 1, 2, 3, 4, 8);
    EXPECT_STREQ("blobname", b.name);
    EXPECT_EQ(1, b.batchSize);
    EXPECT_EQ(2, b.x);
    EXPECT_EQ(3, b.y);
    EXPECT_EQ(4, b.z);
    EXPECT_EQ(8, b.sizeofEle);
}

TEST(BlobTest, init_zero_dim)
{
    Blob b = Blob();
    EXPECT_EQ(ZERO_DIM, b.init());
}

TEST(BlobTest, init__data_exist)
{
    Blob b = Blob("blobname", 1, 2, 3, 4, 8);
    EXPECT_EQ(0, b.init());
    EXPECT_EQ(0, b.init());
}