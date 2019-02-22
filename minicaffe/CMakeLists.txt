project(minicafe)

add_compile_options(-std=c++11)

add_library(blob STATIC blob.cpp)

if(TEST_ENABLED)
    add_subdirectory(test)    
endif(TEST_ENABLED)
