#include "gtest/gtest.h"
#include "../pooling.h"

TEST(PoolingLayerTest, infer_bp)
{
//	printf("\n*********start pooling infer test*********\n");
	int in_x, in_y, in_z, batch_size = 2;
	in_x = 11, in_y = 11, in_z = 3;

	Blob b1 = Blob("b1", batch_size, in_x, in_y, in_z, sizeof(float));
	Blob b_bp = Blob("b_bp", batch_size, in_x, in_y, in_z, sizeof(float));
	b1.init();
	b_bp.init();

	int err = random_matrix(b1._data, b1.get_ele_num());
	ASSERT_EQ(0, err);
/*
	printf("b1.get_ele_num() = %d\n", b1.get_ele_num());

	err = print_matrix(b1._data, batch_size, in_x, in_y, in_z);
	if (err != 0)
	{
		printf("%d\n", err);
	}
	ASSERT_EQ(0, err);
*/
	PoolingLayer testPoolingLayer = PoolingLayer("test");
	vector<Blob*> left_blobs, right_blobs, bp_left_blobs;

	int inputs_dims[4], outputs_dims[4];
	int out_x, out_y, out_z;
	inputs_dims[0] = batch_size, inputs_dims[1] = in_x, inputs_dims[2] = in_y, inputs_dims[3] = in_z;

	testPoolingLayer.get_outputs_dimensions(inputs_dims, 1, outputs_dims, 1);
	out_x = outputs_dims[1], out_y = outputs_dims[2], out_z = outputs_dims[3];
//	int i;
/*
	for (i = 0; i < 4; i++)
	{
		printf("outputs_dims[%d] = %d\n", i, outputs_dims[i]);
	}
*/
	ASSERT_EQ(batch_size, outputs_dims[0]);
	ASSERT_EQ(4, outputs_dims[1]);
	ASSERT_EQ(4, outputs_dims[2]);
	ASSERT_EQ(3, outputs_dims[3]);

	Blob b2 = Blob("b2", batch_size, out_x, out_y, out_z, sizeof(float));
	b2.init();

	left_blobs.push_back(&b1);
	right_blobs.push_back(&b2);
	bp_left_blobs.push_back(&b_bp);

	testPoolingLayer.infer(left_blobs, right_blobs);
/*
	err = print_matrix(right_blobs[0]->_data, batch_size, out_x, out_y, out_z);
	if (err != 0)
	{
		printf("%d\n", err);
	}

	err = print_coord_matrix(testPoolingLayer.tmp_space, batch_size, out_x, out_y, out_z);
	if (err != 0)
	{
		printf("%d\n", err);
	}
*/
	testPoolingLayer.bp(bp_left_blobs, right_blobs);
/*
	err = print_matrix(bp_left_blobs[0]->_data, batch_size, in_x, in_y, in_z);
	if (err != 0)
	{
		printf("%d\n", err);
	}
	printf("\n*********finish pooling infer test*********\n");
*/
}