#ifndef RESIZEIMAGEPARALLEL_H
#define RESIZEIMAGEPARALLEL_H

#include <vector>

enum ConvolutionType {
    NonVectorized,
    Vectorized
};

unsigned char* applyConvolutionCUDA(const unsigned char* input, int width, int height,
                                    const std::vector<std::vector<float>>& kernel,
                                    ConvolutionType type,
                                    int threadsX, int threadsY);

#endif
