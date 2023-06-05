#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nppi.h>

#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
namespace chrono = std::chrono;
using std::chrono::system_clock;
using duration_ms = chrono::duration<double, std::milli>;

// clang-format off
void resizeBilinear(float *d_src, float *d_dest, int W, int H, int B);
void subsample(float *d_src, float *d_dest, int W, int H, int B);
// clang-format on

int main() {

    int B = 8;
    int H = 1080;
    int W = 1920;

    cv::Mat img1 = cv::imread("../Data/test1.png");
    cv::cvtColor(img1, img1, cv::COLOR_BGR2GRAY);
    img1.convertTo(img1, CV_32FC1);
    cv::resize(img1, img1, {W, H});

    cv::Mat img2 = cv::imread("../Data/test2.png");
    cv::cvtColor(img2, img2, cv::COLOR_BGR2GRAY);
    img2.convertTo(img2, CV_32FC1);
    cv::resize(img2, img2, {W, H});

    int n_pixels = H * W;
    int n_bytes = n_pixels * sizeof(float);

    float *img_in;
    cudaMalloc(&img_in, B * n_bytes);
    cudaMemcpy(img_in, img1.data, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(img_in + n_pixels, img2.data, n_bytes, cudaMemcpyHostToDevice);

    float *img_out;
    cudaMalloc(&img_out, B * n_bytes);

    if (true) {
        std::cout << "\nresize bilinear: \n";
        for (int i = 0; i < 10; ++i) {
            cudaDeviceSynchronize();
            system_clock::time_point t1 = system_clock::now();

            resizeBilinear(img_in, img_out, W, H, B);

            cudaDeviceSynchronize();
            system_clock::time_point t2 = system_clock::now();
            duration_ms dur = t2 - t1;
            std::cout << "time (ms): " << dur.count() << "\n";
        }
    }

    if (true) {
        std::cout << "\nsubsample plain: \n";
        for (int i = 0; i < 10; ++i) {
            cudaDeviceSynchronize();
            system_clock::time_point t1 = system_clock::now();

            subsample(img_in, img_out, W, H, B);

            cudaDeviceSynchronize();
            system_clock::time_point t2 = system_clock::now();
            duration_ms dur = t2 - t1;
            std::cout << "time (ms): " << dur.count() << "\n";
        }
    }

    if (true) {
        std::cout << "\nNPPI: \n";
        for (int i = 0; i < 10; ++i) {
            cudaDeviceSynchronize();
            system_clock::time_point t1 = system_clock::now();

            for (int b = 0; b < 8; ++b) {
                nppiResize_32f_C1R(
                    img_in,
                    W * sizeof(float),
                    {W, H},
                    {0, 0, W, H},
                    img_out,
                    W / 4 * sizeof(float),
                    {W / 4, H / 4},
                    {0, 0, W / 4, H / 4},
                    NPPI_INTER_NN);
            }

            cudaDeviceSynchronize();
            system_clock::time_point t2 = system_clock::now();
            duration_ms dur = t2 - t1;
            std::cout << "time (ms): " << dur.count() << "\n";
        }
    }

    cudaMemcpy(img1.data, img_out, n_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(img2.data, img_out + n_pixels, n_bytes, cudaMemcpyDeviceToHost);

    cv::imwrite("../Data/test1_post.png", img1);
    cv::imwrite("../Data/test2_post.png", img2);
    return 0;
}
