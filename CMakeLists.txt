cmake_minimum_required(VERSION 3.24)
project(lb2)

add_executable(split_text_simple src/split_text_simple.cpp src/split_text_common.cpp)
add_executable(embedding src/embedding.cpp)
