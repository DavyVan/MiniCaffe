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

file(GLOB MINICAFE_SRC_FILES "*.cpp")
add_library(minicaffe STATIC ${MINICAFE_SRC_FILES})