# UCR-CS217-FinalProject

master: [![Build Status](https://travis-ci.com/DavyVan/UCR-CS217-FinalProject.svg?token=aPmAPRxERpUR8kmR2XzD&branch=master)](https://travis-ci.com/DavyVan/UCR-CS217-FinalProject)
fq-dev: [![Build Status](https://travis-ci.com/DavyVan/UCR-CS217-FinalProject.svg?token=aPmAPRxERpUR8kmR2XzD&branch=fq-dev)](https://travis-ci.com/DavyVan/UCR-CS217-FinalProject)

## NOTICE

* Add wrapper for each layer. For example: FullyConnectedLayer has a wrapper fc(Net* net, name, lefts, right[, _layer specific params_])
* Call base class Layer's constructor function with layer's name

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
