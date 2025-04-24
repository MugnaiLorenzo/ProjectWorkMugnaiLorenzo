#ifndef KERNEL_H
#define KERNEL_H

#include <vector>
#include <string>

class Kernel {
public:
    Kernel();
    std::vector<std::pair<std::vector<std::vector<float>>, std::string>> getKernels();

private:
    std::vector<std::pair<std::vector<std::vector<float>>, std::string>> kernels;
};

#endif
