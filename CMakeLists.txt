cmake_minimum_required(VERSION 3.30)
project(ProjectWorkMugnaiLorenzo CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 17)

add_executable(ProjectWorkMugnaiLorenzo main.cu
        Kernel.cpp
        Kernel.h
        stb_image.h
        stb_image_write.h
        ResizeImage.cpp
        ResizeImage.h
        ResizeImageParallel.cu
        ResizeImageParallel.cuh)

set_target_properties(ProjectWorkMugnaiLorenzo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
