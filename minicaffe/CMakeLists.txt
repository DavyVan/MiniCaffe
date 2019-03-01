project(minicafe)

add_compile_options(-std=c++11)

file(GLOB MINICAFE_SRC_FILES "*.cpp")
add_library(minicafe STATIC ${MINICAFE_SRC_FILES})

if(TEST_ENABLED)
    add_subdirectory(test)    
endif(TEST_ENABLED)
