# UCR-CS217-FinalProject

master: [![Build Status](https://travis-ci.com/DavyVan/UCR-CS217-FinalProject.svg?token=aPmAPRxERpUR8kmR2XzD&branch=master)](https://travis-ci.com/DavyVan/UCR-CS217-FinalProject)
fq-dev: [![Build Status](https://travis-ci.com/DavyVan/UCR-CS217-FinalProject.svg?token=aPmAPRxERpUR8kmR2XzD&branch=fq-dev)](https://travis-ci.com/DavyVan/UCR-CS217-FinalProject)

## How to build MiniCaffe

<code>

    cd [root_dir]/minicaffe/build  
    cmake ..            # Without test
    cmake --build .     # Then find target library in minicaffe/build/
</code>

## How to build with test

<code>

    cd [root_dir]/minicaffe/build
    cmake .. -DTEST_ENABLED=ON      # This option will enable GoogleTest and our test case
    cmake --build .                 # Then find target library in minicaffe/build/ 
                                      and xxx_test in minicaffe/build/test
</code>

## How to build main.cpp

<code>

    cd [root_dir]/build
    cmake ..
    cmake --build .     # Then find the final executable in [root dir]/build/
</code>