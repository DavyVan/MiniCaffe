#include <cstring>
#include "seqnet.h"
#include "errors.h"
#include "layer.h"

int SeqNet::batchsize = 32;

SeqNet::SeqNet(int batchsize)
{
    dataGenerator = NULL;
    this->batchsize = batchsize;
}

SeqNet::SeqNet(MnistGenerator* generator, int batchsize)
{
    dataGenerator = generator;
    this->batchsize = batchsize;
}

void SeqNet::update_generator(MnistGenerator* generator)
{
    dataGenerator = generator;
}

int SeqNet::add_layer(Layer* layer, const char* lefts[], const int numLefts, const char* rights[], const int numRights)
{
    // Check lefts to make sure all of them are in this net.
    vector<Blob*> leftBlobExisted;      // the left blob to be connect with current layer;
    vector<Blob*> rightBlobsCreated;    // the right blobs created

    // For each left in lefts[], do checks. It's OK if numLefts is zero.
    bool isOK = true;
    int numBlobs = blobs.size();
    for (int iLefts = 0; iLefts < numLefts; iLefts++)
    {
        int idx = get_blob_id_by_name(lefts[iLefts]);
        if (idx != -1)
        {
            leftBlobExisted.push_back(blobs[idx]);
        }
        else
        {
            isOK = false;
        }
    }
    // If not match any, then abort.
    if (isOK == false && numLefts != 0) // No match
    {
        print_err_str(LEFT_NOT_MATCH);
        return LEFT_NOT_MATCH;
    }

    // No check for repeated names in rights. We will be sure no name can show up twice manually.

    // Check passed.

    // Decide right blobs dimensions.
    int* inputDims = new int[numLefts * 4];
    int* outputDims = new int[numRights * 4];
    for (int i = 0; i < numLefts; i++)
    {
        inputDims[i * 4 + 0] = leftBlobExisted[i]->batchSize;
        inputDims[i * 4 + 1] = leftBlobExisted[i]->x;
        inputDims[i * 4 + 2] = leftBlobExisted[i]->y;
        inputDims[i * 4 + 3] = leftBlobExisted[i]->z;
    }
    layer->get_outputs_dimensions(inputDims, numLefts, outputDims, numRights);

    // Create right blob instances
    for (int i = 0; i < numRights; i++)     // for each right blob
    {
        // create
        Blob *b = new Blob(rights[i], outputDims[i*4 + 0], outputDims[i*4 + 1], outputDims[i*4 + 2], outputDims[i*4 + 3], 4);
        // add new blob to net::blobs
        blobs.push_back(b);
        // add new blob to right list
        rightBlobsCreated.push_back(b);
    }

    // add current layer to net
    layers.push_back(layer);

    // Add left/right vector to net, no matter they are empty or not.
    this->lefts.push_back(leftBlobExisted);
    this->rights.push_back(rightBlobsCreated);

    // delete inputDims, outputDims
    delete inputDims;
    delete outputDims;
    return 0;
}

int SeqNet::init()
{
    int err = 0;

    // TODO: Check dimensions. The function in each layer is not implemented yet.

    // Call init() of all blobs
    for (std::vector<Blob*>::iterator it = blobs.begin(); it != blobs.end(); it++)
        err += (*it)->init();

    // Call init() of all layers
    for (std::vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++)
        err += (*it)->init();

    return err;
}

void SeqNet::infer(bool gpu_enabled)
{
    // Reset all blobs
    for (std::vector<Blob*>::iterator it = blobs.begin(); it != blobs.end(); it++)
        (*it)->reset();
    
    // Fetch a new batch of samples from Generator
    if (dataGenerator == NULL)
    {
        printf("No data generator assigned.\n");
        return;
    }
    std::vector<Blob> generated = dataGenerator->loadSample(batchsize);
    std::vector<Blob*> generated_ptr;
    for (std::vector<Blob>::iterator it = generated.begin(); it != generated.end(); it++)
        generated_ptr.push_back(new Blob(*it));
    lefts[0] = generated_ptr;

    int n = layers.size();
    for (int i = 0; i < n; i++)
    {
        if (gpu_enabled)
            layers[i]->infer_gpu(lefts[i], rights[i]);
        else
            layers[i]->infer(lefts[i], rights[i]);
    }
}

void SeqNet::bp(bool gpu_enabled)
{
    int n = layers.size();
    for (int i = n-1; i >= 0; i--)
    {
        if (gpu_enabled)
            layers[i]->bp_gpu(lefts[i], rights[i]);
        else
            layers[i]->bp(lefts[i], rights[i]);
    }
}

void SeqNet::train(int iteration, bool gpu_enabled)
{
    for (int i = 0; i < iteration; i++)
    {
        infer(gpu_enabled);
        bp(gpu_enabled);
        Blob* loss = get_output("loss");
        printf("Training: batch-%d, loss=%f\n", i, loss->_data[0]);
    }

}

Blob* SeqNet::get_output(const char* name)
{
    int idx = get_blob_id_by_name(name);
    if (idx == -1)
    {
        printf("Blob not found.\n");
        return NULL;
    }
    
    Blob* ret = new Blob(*blobs[idx]);
    return ret;
} 

int SeqNet::get_batchsize()
{
    return batchsize;
}

int SeqNet::get_blob_id_by_name(const char *name)
{
    int numBlobs = blobs.size();

    for (int iBlobs = 0; iBlobs < numBlobs; iBlobs++)
    {
        if (strcmp(blobs[iBlobs]->name, name) == 0)
        {
            return iBlobs;
        }
    }
    return -1;
}