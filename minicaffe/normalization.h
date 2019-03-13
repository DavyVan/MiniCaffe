#ifndef _NL_H_
#define _NL_H_

#include <cstdlib>
//#include "net.h"
#include "layer.h"
#include "errors.h"
#include "util.h"
#include <string.h>

class NormalizationLayer : public Layer
{
public:
	int nn;
	float alpha;
	float beta;
	float kk;
	
	NormalizationLayer();
	NormalizationLayer(char *name);
	NormalizationLayer(char *name, float alpha_, float beta_, float kk_, int nn_);

	~NormalizationLayer();

	int init();

	void infer(vector<Blob*> left_blobs, vector<Blob*> right_blobs);

	void bp(vector<Blob*> left_blobs, vector<Blob*> right_blobs);

	void get_outputs_dimensions(int inputs_dims[], const int numInputs, int outputs_dims[], const int numOutputs);

	bool check_dimensions();
};

#endif
