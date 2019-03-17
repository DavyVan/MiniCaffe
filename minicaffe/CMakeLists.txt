project(minicaffe)

add_compile_options(-std=c++11)


if(TEST_ENABLED)
    add_subdirectory(test)
    if(MSVC)
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /D TEST_ENABLED")
    else()
        add_compile_options(-DTEST_ENABLED)
    endif()
    include_directories(third_party/googletest/googletest/include)
endif(TEST_ENABLED)

#file(GLOB LAYERS_FILES "layers/*.cpp")
#file(GLOB MINICAFE_SRC_FILES "*.cpp")
#set(MINICAFE_SRC_FILES ${MINICAFE_SRC_FILES} ${LAYERS_FILES})
#add_library(minicaffe STATIC ${MINICAFE_SRC_FILES})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
file(GLOB MINICAFE_SRC_FILES "layers/*.cpp" "layers/*.cu" "*.cpp" "*.cu")

set(CUDA_NVCC_FLAGS "-g -G --std=c++11")
if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    endif ()
else ()
    set (CMAKE_CXX_STANDARD 11)
endif ()

set(PLATFORM -gencode=arch=compute_61,code=compute_61)
set(PLATFORM ${PLATFORM} -gencode arch=compute_61,code=sm_61)
set(PLATFORM ${PLATFORM} -gencode arch=compute_60,code=sm_60)
set(PLATFORM ${PLATFORM} -gencode arch=compute_52,code=sm_52)
set(PLATFORM ${PLATFORM} -gencode arch=compute_50,code=sm_50)
set(PLATFORM ${PLATFORM} -gencode arch=compute_35,code=sm_35)
set(PLATFORM ${PLATFORM} -gencode arch=compute_30,code=sm_30)

cuda_add_library(minicaffe ${MINICAFE_SRC_FILES} OPTIONS )