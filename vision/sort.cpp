// #include "box_filter.cuh"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
namespace chrono = std::chrono;
using duration_ms = chrono::duration<double, std::milli>;

// clang-format off
void raidxSortSingle(float *img_in, float *img_out, int B, int H, int W);
void raidxSortBatch(float *img_in, float *img_out, int B, int H, int W);
// clang-format on

int main() {

    int B = 8;
    int H = 1080 / 4;
    int W = 1920 / 4;

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
        std::cout << "radixSortSingle\n";
        raidxSortSingle(img_in, img_out, B, H, W);
    }

    if (true) {
        std::cout << "radixSortBatch\n";
        raidxSortBatch(img_in, img_out, B, H, W);
    }

    cudaMemcpy(img1.data, img_out, n_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(img2.data, img_out + n_pixels, n_bytes, cudaMemcpyDeviceToHost);

    cv::imwrite("../Data/test1_post.png", img1);
    cv::imwrite("../Data/test2_post.png", img2);
    return 0;
}
