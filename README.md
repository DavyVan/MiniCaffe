# UCR-CS217-FinalProject

master: [![Build Status](https://travis-ci.com/DavyVan/UCR-CS217-FinalProject.svg?token=aPmAPRxERpUR8kmR2XzD&branch=master)](https://travis-ci.com/DavyVan/UCR-CS217-FinalProject)
fq-dev: [![Build Status](https://travis-ci.com/DavyVan/UCR-CS217-FinalProject.svg?token=aPmAPRxERpUR8kmR2XzD&branch=fq-dev)](https://travis-ci.com/DavyVan/UCR-CS217-FinalProject)

## NOTICE

1. __How to compute:__ As shown in layer.h, necessary data are already stored in the base class. So no more arguments for `infer()` and `bp()`. 
1. __Add wrapper for each layer.__ For example: FullyConnectedLayer has a wrapper `FullyConnectedLayer FullyConnectedLayer::fc(Net* net, name, lefts, right[, layer specific params])`. You should call `Layer::add_to_net()` in the wrapper, so you need some extra arguments. Please refer to `layer.h`. Then, you should check the return value of `Layer::add_to_net()` to make sure everything was ok, if any error occurred the program may abort (use `exit(errno)`). 
2. __Layer Class:__ Call base class Layer's constructor function with layer's name
3. __Must provide `get_outputs_dimensions()`.__ Each blob has 4 dimensions \[batchsize, x, y, z\]. If exist more than one blobs, concatenate them in one integer array.
5. __Error Handling:__ If you need to tell caller that something went wrong, you can return a error code. You can define your own error code in `errors.h` (pick next available number) and add its human readable error message in `errors.cpp`.
1. __Encounter some open problems?__ You can label things to be done by add a comment begin with `//TODO:` (case sensitive) so that all unsolved problem can be tracked by some IDEs.
1. __Do no need to modify `CMakeLists.txt` to track new files.__ I have already modified it to enable cmake to automatically search all *.cpp files and include them for compiling.

### Limitations
1. Simplified network construction: A blob can only be modified AND used by one layer. Only input layer and loss layer (the final one) can have multiple outputs/inputs and the order of blobs must be pre-defined manually. (_e.g._ input layer produces two blobs "data" and "label", not "label" and "data").
2. No check for repeated names of blobs and layers.
1. For other unstated limitations, please see the TODOs in source code. 

## How to build MiniCaffe

    git clone [url]  
    cd [root_dir]  
    git submodule init  
    git submodule update
    cd minicaffe/build  
    cmake ..            # Without test
    cmake --build .     # Then find target library in minicaffe/build/ 


## How to build with test

    git clone [url]
    cd [root_dir]
    git submodule init
    git submodule update
    cd minicaffe/build
    cmake .. -DTEST_ENABLED=ON      # This option will enable GoogleTest and our test case
    cmake --build .                 # Then find target library in minicaffe/build/ 
                                      and xxx_test in minicaffe/build/test

## How to build main.cpp

    git clone [url]
    cd [root_dir]
    git submodule init
    git submodule update
    cd build
    cmake ..
    cmake --build .     # Then find the final executable in [root dir]/build/
