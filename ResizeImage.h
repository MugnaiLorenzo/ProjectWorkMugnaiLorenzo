#ifndef RESIZEIMAGE_H
#define RESIZEIMAGE_H

#include <vector>

class ResizeImage {
public:
    ResizeImage(unsigned char* input, int width, int height,
                const std::vector<std::vector<float>>& kernel);

    unsigned char* transform();  // restituisce immagine convoluta (malloc/free gestito da utente)

private:
    unsigned char* img_in;
    int w, h;
    std::vector<std::vector<float>> kernel;
    int ksize;
    int offset;

    unsigned char applyKernel(int x, int y, int c);
};

#endif
