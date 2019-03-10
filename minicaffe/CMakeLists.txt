project(minicaffe)

add_compile_options(-std=c++11)

file(GLOB LAYERS_FILES "layers/*.cpp")
file(GLOB MINICAFE_SRC_FILES "*.cpp")
set(MINICAFE_SRC_FILES ${MINICAFE_SRC_FILES} ${LAYERS_FILES})
add_library(minicaffe STATIC ${MINICAFE_SRC_FILES})
add_library(blob STATIC ${MINICAFE_SRC_FILES})
if(TEST_ENABLED)
    add_subdirectory(test)    
endif(TEST_ENABLED)
