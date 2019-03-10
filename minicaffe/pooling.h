#ifndef _NL_H_
#define _NL_H_
	
#include <cstdlib>
#include "net.h"
#include "layer.h"
#include "errors.h"
	
class PoolingLayer : public Layer
{
public:
	int mask_x;
	int mask_y;
	int stride;
	
	PoolingLayer();
	PoolingLayer(char *name);
	PoolingLayer(char *name, int mask_x_, int mask_y_, int stride_);
	~PoolingLayer();
	
	int init();
	void infer();
	void bp();
	void get_outputs_dimensions(int inputs_dims[], const int numInputs, int outputs_dims[], const int numOutputs);
	bool check_dimensions();
};
	
#endif
