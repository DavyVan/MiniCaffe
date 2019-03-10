cmake_minimum_required(VERSION 2.8.11)

project(minicaffe_test)
add_compile_options(-std=c++11)
execute_process(COMMAND mkdir build
  RESULT_VARIABLE result
  WORKING_DIRECTORY ../third_party/googletest/googletest )
# execute_process(COMMAND ${CMAKE_COMMAND} .. -DCMAKE_CXX_FLAGS=-std=c++11 -G "${CMAKE_GENERATOR}"
#   RESULT_VARIABLE result
#   WORKING_DIRECTORY ../third_party/googletest/googletest/build )
# if(result)
#   message(FATAL_ERROR "CMake step for googletest failed: ${result}")
# endif()
# execute_process(COMMAND ${CMAKE_COMMAND} --build build
#   RESULT_VARIABLE result
#   WORKING_DIRECTORY ../third_party/googletest/googletest )
# if(result)
#   message(FATAL_ERROR "Build step for googletest failed: ${result}")
# endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(../third_party/googletest/googletest
                 ../third_party/googletest/googletest/build
                 EXCLUDE_FROM_ALL)

file(GLOB TEST_SRC_FILES "*.cpp")
add_executable(test ${TEST_SRC_FILES})
target_link_libraries(test minicaffe gtest_main)

# include(GoogleTest)
# gtest_add_tests(TARGET "blob_test")   # We do not use CTest to manage test cases.