#include "ResizeImage.h"
#include <algorithm>
#include <cstring>
#include <algorithm>

ResizeImage::ResizeImage(unsigned char *input, int width, int height,
                         const std::vector<std::vector<float> > &kernel): img_in(input), w(width), h(height),
                                                                          kernel(kernel) {
    ksize = static_cast<int>(kernel.size());
    offset = ksize / 2;
}

unsigned char ResizeImage::applyKernel(int x, int y, int c) {
    float acc = 0.0f;
    for (int i = -offset; i <= offset; ++i) {
        for (int j = -offset; j <= offset; ++j) {
            int nx = x + j;
            int ny = y + i;
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            int idx = (ny * w + nx) * 3 + c;
            acc += img_in[idx] * kernel[i + offset][j + offset];
        }
    }
    return static_cast<unsigned char>(std::clamp(acc, 0.0f, 255.0f));
}

unsigned char *ResizeImage::transform() {
    unsigned char *output = new unsigned char[w * h * 3];
    std::memset(output, 0, w * h * 3);

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c)
                output[(y * w + x) * 3 + c] = applyKernel(x, y, c);

    return output;
}
