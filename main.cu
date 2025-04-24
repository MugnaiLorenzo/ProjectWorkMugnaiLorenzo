#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <unordered_map>
#include <vector>
#include "Kernel.h"
#include "ResizeImage.h"
#include "ResizeImageParallel.cuh"

namespace fs = std::filesystem;
using namespace std::chrono;

int main() {
    Kernel kernelGen;
    auto kernels = kernelGen.getKernels();
    std::vector<std::string> resolutions = {"480x270", "960x540", "1920x1080", "4k", "8k"};
    std::vector<int> thread_counts = {2, 4, 8, 16};

    std::ofstream csv("../times.csv");
    csv << "Resolution,Type,Threads,Time(s),SpeedUp\n";

    for (const auto &res: resolutions) {
        //CARICAMENTO DELLE CARTELLE
        std::string folder = "../image/" + res;
        if (!fs::exists(folder)) continue;
        std::vector<std::string> images;
        for (const auto &entry: fs::directory_iterator(folder))
            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")
                images.push_back(entry.path().string());
        if (images.empty()) continue;
        fs::create_directories("../output_seq/" + res);
        fs::create_directories("../output_cuda/" + res);
        std::cout << "\n== RISOLUZIONE: " << res << " ==\n";

        std::unordered_map<std::string, double> times;
        times[res] = 0;
        for (int threads: thread_counts) {
            times[res + std::to_string(threads)] = 0;
            times[res + std::to_string(threads) + "Vect"] = 0;
        }
        for (const auto &[kernelMatrix, kernelName]: kernels) {
            for (size_t i = 0; i < images.size(); ++i) {
                int w, h, ch;
                unsigned char *img = stbi_load(images[i].c_str(), &w, &h, &ch, 3);
                std::string name = fs::path(images[i]).stem().string();
                std::string out_path_seq = "../output_seq/" + res + "/" + name + kernelName + ".png";
                std::string out_path_cuda = "../output_cuda/" + res + "/" + name + kernelName + ".png";
                if (!img) continue;
                //SEQUENZIALE
                auto start_k = high_resolution_clock::now();
                ResizeImage resizer(img, w, h, kernelMatrix);
                unsigned char *out = resizer.transform();
                times[res] += duration<double>(high_resolution_clock::now() - start_k).count();
                stbi_write_png(out_path_seq.c_str(), w, h, 3, out, w * 3);
                delete[] out;
                for (int threads: thread_counts) {
                    //NON VETTORIZZATO
                    start_k = high_resolution_clock::now();
                    unsigned char *out_par = applyConvolutionCUDA(img, w, h, kernelMatrix, NonVectorized, threads,
                                                                  threads);
                    times[res + std::to_string(threads)] += duration<double>(high_resolution_clock::now() - start_k).
                            count();
                    //VETTORIZZATO
                    start_k = high_resolution_clock::now();
                    unsigned char *out_par_vect = applyConvolutionCUDA(img, w, h, kernelMatrix, Vectorized, threads,
                                                                       threads);
                    times[res + std::to_string(threads) + "Vect"] += duration<double>(
                        high_resolution_clock::now() - start_k).count();
                    if (threads == 2) {
                        stbi_write_png(out_path_cuda.c_str(), w, h, 3, out_par, w * 3);
                    }
                    delete[] out_par;
                    delete[] out_par_vect;
                }
                stbi_image_free(img);
            }
        }
        std::cout << "RISOLZUIONE RES: " << res << " TEMPO SEQUEZIALE: " << times[res] << "s\n";
        csv << res << ",CPU,0," << times[res] << ",1" << "\n";
        for (int threads: thread_counts) {
            double speed = times[res] / times[res + std::to_string(threads)];
            double speed_vect = times[res] / times[res + std::to_string(threads) + "Vect"];
            std::cout << "RISOLZUIONE RES: " << res << " TEMPO NON VETTORIZZATO con: " << threads << " THREADS: " <<
                    times[
                        res + std::to_string(threads)] << "s" << " SPEED UP: " << speed << "\n";
            std::cout << "RISOLZUIONE RES: " << res << " TEMPO VETTORIZZATO con: " << threads << " THREADS: " << times[
                res + std::to_string(threads) + "Vect"] << "s" << " SPEED UP: " << speed_vect << "\n";
            csv << res << ",CUDA_NonVectorized," << threads << "," << times[res + std::to_string(threads)] << "," <<
                    speed << "\n";
            csv << res << ",CUDA_Vectorized," << threads << "," << times[res + std::to_string(threads) + "Vect"] << ","
                    << speed_vect << "\n";
        }
    }

    csv.close();
    std::cout << "\n== Fine! Tempi salvati in times.csv ==\n";
    return 0;
}
