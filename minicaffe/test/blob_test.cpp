#include "gtest/gtest.h"
#include "../blob.h"

TEST(BlobTest, get_ele_num_zero)
{
    Blob blob = Blob();
    blob.x = blob.y = blob.z = 0;

    ASSERT_EQ(0, blob.get_ele_num());
}

TEST(BlobTest, get_ele_num_nonzero)
{
    Blob blob = Blob();
    blob.x = 1;
    blob.y = 2;
    blob.z = 3;

    ASSERT_EQ(6, blob.get_ele_num());
}
