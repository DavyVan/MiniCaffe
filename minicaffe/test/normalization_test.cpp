#include "gtest/gtest.h"
#include "../normalization.h"

#define FLOAT_DIFF 1.1e-11

TEST(NormalizationLayerTest, one_param_constructor)
{
	NormalizationLayer testNormalizationLayer = NormalizationLayer("test");
	EXPECT_STREQ("test", testNormalizationLayer.name);
	ASSERT_EQ(5, testNormalizationLayer.nn);
	ASSERT_NEAR(2.0, testNormalizationLayer.kk, FLOAT_DIFF);
	ASSERT_NEAR(1e-4, testNormalizationLayer.alpha, FLOAT_DIFF);
	ASSERT_NEAR(0.75, testNormalizationLayer.beta, FLOAT_DIFF);
}

TEST(NormalizationLayerTest, mul_param_constructor)
{
	int nn_ = 2;
	float kk_ = 3.0, alpha_ = 1e-5, beta_ = 0.8;
	NormalizationLayer testNormalizationLayer = NormalizationLayer("test", alpha_, beta_, kk_, nn_);
	EXPECT_STREQ("test", testNormalizationLayer.name);
	ASSERT_EQ(nn_, testNormalizationLayer.nn);
	ASSERT_NEAR(kk_, testNormalizationLayer.kk, FLOAT_DIFF);
	ASSERT_NEAR(alpha_, testNormalizationLayer.alpha, FLOAT_DIFF);
	ASSERT_NEAR(beta_, testNormalizationLayer.beta, FLOAT_DIFF);
}

TEST(NormalizationLayerTest, infer_bp)
{
//	printf("\n*********start pooling infer test*********\n");
	int in_x, in_y, in_z, batch_size = 2;
	in_x = 11, in_y = 11, in_z = 3;

	Blob b1 = Blob("b1", batch_size, in_x, in_y, in_z, sizeof(float));
	
	b1.init();
	int err = random_matrix(b1._data, b1.get_ele_num());
	ASSERT_EQ(0, err);

	Blob b_bp = b1;

//	printf("b1.get_ele_num() = %d\n", b1.get_ele_num());
	memcpy(b_bp._data, b1._data, b1.get_ele_num() * sizeof(float));
/*
	err = print_matrix(b1._data, batch_size, in_x, in_y, in_z);
	if (err != 0)
	{
		printf("%d\n", err);
	}
	ASSERT_EQ(0, err);

	err = print_matrix(b_bp._data, batch_size, in_x, in_y, in_z);
	if (err != 0)
	{
		printf("%d\n", err);
	}
	ASSERT_EQ(0, err);
*/
	NormalizationLayer testNormalizationLayer = NormalizationLayer("test");
	vector<Blob*> left_blobs, right_blobs, bp_left_blobs;

	int inputs_dims[4], outputs_dims[4];
	int out_x, out_y, out_z;
	inputs_dims[0] = batch_size, inputs_dims[1] = in_x, inputs_dims[2] = in_y, inputs_dims[3] = in_z;

	testNormalizationLayer.get_outputs_dimensions(inputs_dims, 1, outputs_dims, 1);
	out_x = outputs_dims[1], out_y = outputs_dims[2], out_z = outputs_dims[3];
/*	int i;

	for (i = 0; i < 4; i++)
	{
		printf("outputs_dims[%d] = %d\n", i, outputs_dims[i]);
	}
*/
	ASSERT_EQ(inputs_dims[0], outputs_dims[0]);
	ASSERT_EQ(inputs_dims[1], outputs_dims[1]);
	ASSERT_EQ(inputs_dims[2], outputs_dims[2]);
	ASSERT_EQ(inputs_dims[3], outputs_dims[3]);

	Blob b2 = Blob("b2", batch_size, out_x, out_y, out_z, sizeof(float));
	b2.init();

	left_blobs.push_back(&b1);
	right_blobs.push_back(&b2);
	bp_left_blobs.push_back(&b_bp);

	testNormalizationLayer.infer(left_blobs, right_blobs);
/*
	err = print_matrix(right_blobs[0]->_data, batch_size, out_x, out_y, out_z);
	if (err != 0)
	{
		printf("%d\n", err);
	}

*/
	testNormalizationLayer.bp(bp_left_blobs, right_blobs);
/*
	err = print_matrix(bp_left_blobs[0]->_data, batch_size, in_x, in_y, in_z);
	if (err != 0)
	{
		printf("%d\n", err);
	}
	printf("\n*********finish pooling infer test*********\n");
*/
}
