cmake_minimum_required(VERSION 3.13)

project(TensorflowSavedModelExample)

set(CMAKE_CXX_STANDARD 14)

add_executable(load_example load_example.cc mnist.h mnist.cc)

# Needs symlink libtensorflow_cc.so -> libtensorflow_cc.so.2
# Means that you have to bazel build libtensorflow_cc.so, not tensorflow_cc
target_link_libraries(load_example tensorflow_cc)
target_link_libraries(load_example tensorflow_framework)
target_link_directories(load_example PRIVATE /usr/local/google/home/bmzhao/Code/tensorflow/bazel-bin/tensorflow)

target_compile_options(load_example PRIVATE -Wl,-rpath=/usr/local/google/home/bmzhao/Code/tensorflow/bazel-bin/tensorflow)

target_include_directories(load_example PRIVATE /usr/local/google/home/bmzhao/Code/tensorflow/bazel-bin/tensorflow/include)
# Need to include this, otherwise protobuf's header  #include <google/protobuf/port_def.inc> is not found
target_include_directories(load_example PRIVATE /usr/local/google/home/bmzhao/Code/tensorflow/bazel-bin/tensorflow/include/src)
