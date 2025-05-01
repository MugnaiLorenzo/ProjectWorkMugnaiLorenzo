#include "ResizeImageParallel.cuh"
#include <cuda_runtime.h>
#include <iostream>

__constant__ float d_kernel_const[5 * 5];

__device__ float clamp(float val) {
    return val < 0 ? 0 : (val > 255 ? 255 : val);
}

// Kernel non vettoriale: 1 thread per canale
__global__ void convolveKernel_NonVec(unsigned char* input, unsigned char* output,
                                      int width, int height, int ksize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = threadIdx.z;  // canale RGB

    int offset = ksize / 2;
    if (x >= width || y >= height || c >= 3) return;

    float acc = 0.0f;
    for (int i = -offset; i <= offset; ++i) {
        for (int j = -offset; j <= offset; ++j) {
            int nx = x + j;
            int ny = y + i;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int idx = (ny * width + nx) * 3 + c;
                acc += input[idx] * d_kernel_const[(i + offset) * ksize + (j + offset)];
            }
        }
    }
    output[(y * width + x) * 3 + c] = static_cast<unsigned char>(clamp(acc));
}

// Kernel vettoriale: 1 thread per pixel, elabora tutti e 3 i canali
__global__ void convolveKernel_Vec(unsigned char* input, unsigned char* output,
                                   int width, int height, int ksize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int offset = ksize / 2;
    if (x >= width || y >= height) return;

    for (int c = 0; c < 3; ++c) {
        float acc = 0.0f;
        for (int i = -offset; i <= offset; ++i) {
            for (int j = -offset; j <= offset; ++j) {
                int nx = x + j;
                int ny = y + i;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int idx = (ny * width + nx) * 3 + c;
                    acc += input[idx] * d_kernel_const[(i + offset) * ksize + (j + offset)];
                }
            }
        }
        output[(y * width + x) * 3 + c] = static_cast<unsigned char>(clamp(acc));
    }
}

unsigned char* applyConvolutionCUDA(const unsigned char* input, int width, int height,
                                    const std::vector<std::vector<float>>& kernel2D,
                                    ConvolutionType type,
                                    int threadsX, int threadsY) {
    int imgSize = width * height * 3;
    int ksize = static_cast<int>(kernel2D.size());
    int klen = ksize * ksize;

    std::vector<float> kernel1D(klen);
    for (int i = 0; i < ksize; ++i)
        for (int j = 0; j < ksize; ++j)
            kernel1D[i * ksize + j] = kernel2D[i][j];

    unsigned char* d_input;
    unsigned char* d_output;

    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);

    cudaMemcpy(d_input, input, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel_const, kernel1D.data(), sizeof(float) * klen);

    dim3 block(threadsX, threadsY);
    dim3 grid((width + threadsX - 1) / threadsX, (height + threadsY - 1) / threadsY);

    if (type == NonVectorized) {
        dim3 block3D(threadsX, threadsY, 3);  // un thread per canale RGB
        convolveKernel_NonVec<<<grid, block3D>>>(d_input, d_output, width, height, ksize);
    } else {
        convolveKernel_Vec<<<grid, block>>>(d_input, d_output, width, height, ksize);
    }

    unsigned char* result = new unsigned char[imgSize];
    cudaMemcpy(result, d_output, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}
