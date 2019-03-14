project(main)

add_subdirectory(minicaffe)     # We now compile the library with main.cpp. May change after installation is implemented (after milestone 2?)

add_executable(main main.cpp)
target_link_libraries(main minicaffe)

add_executable(mc_test mc_test.cpp)
target_link_libraries(mc_test minicaffe)