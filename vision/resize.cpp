#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nppi.h>

#include <cuda_runtime.h>

#include <fmt/core.h>

#include <chrono>
#include <iostream>
namespace chrono = std::chrono;
using std::chrono::system_clock;
using duration_ms = chrono::duration<double, std::milli>;

// clang-format off
void resizeNearest(float *d_src, float *d_dst, int B, int src_H, int src_W, int dst_H, int dst_W);
void resizeBilinear(float *d_src, float *d_dst, int B, int src_H, int src_W, int dst_H, int dst_W);
// clang-format on

int main() {

    int B = 8;
    int src_H = 1080 / 1.5;
    int src_W = 1920 / 1.5;
    int dst_H = src_H * 1.5;
    int dst_W = src_W * 1.5;
    bool bilinear = true;

    std::vector<cv::Mat> imgs;
    for (int i = 0; i < B; ++i) {
        imgs.push_back(cv::imread(fmt::format("../Data/test{}.png", i)));
        cv::cvtColor(imgs[i], imgs[i], cv::COLOR_BGR2GRAY);
        imgs[i].convertTo(imgs[i], CV_32FC1);
        cv::resize(imgs[i], imgs[i], {src_W, src_H});
    }

    int n_pixels = src_H * src_W;
    int n_bytes = n_pixels * sizeof(float);

    float *img_in;
    cudaMalloc(&img_in, B * n_bytes);
    for (int b = 0; b < B; ++b) {
        cudaMemcpy(img_in + b * n_pixels, imgs[b].data, n_bytes, cudaMemcpyHostToDevice);
    }

    int n_pixels_dst = dst_H * dst_W;
    int n_bytes_dst = n_pixels_dst * sizeof(float);

    float *img_out;
    cudaMalloc(&img_out, B * n_bytes_dst);

    if (false) {
        std::cout << "\nopencv cpu: \n";
        for (int b = 0; b < B; ++b) {
            cv::Mat img_out_cv;
            cv::resize(
                imgs[b],
                img_out_cv,
                {dst_W, dst_H},
                0,
                0,
                bilinear ? cv::INTER_LINEAR : cv::INTER_NEAREST_EXACT);
            cudaMemcpy(
                img_out + b * n_pixels_dst,
                img_out_cv.data,
                n_bytes_dst,
                cudaMemcpyHostToDevice);
        }
    }

    if (true) {
        std::cout << "\npure cuda: \n";
        for (int i = 0; i < 10; ++i) {
            cudaDeviceSynchronize();
            system_clock::time_point t1 = system_clock::now();

            if (bilinear) {
                resizeBilinear(img_in, img_out, B, src_H, src_W, dst_H, dst_W);
            } else {
                resizeNearest(img_in, img_out, B, src_H, src_W, dst_H, dst_W);
            }

            cudaDeviceSynchronize();
            system_clock::time_point t2 = system_clock::now();
            duration_ms dur = t2 - t1;
            std::cout << "time (ms): " << dur.count() << "\n";
        }
    }

    if (false) {
        std::cout << "\nNPPI: \n";
        for (int i = 0; i < 10; ++i) {
            cudaDeviceSynchronize();
            system_clock::time_point t1 = system_clock::now();

            for (int b = 0; b < B; ++b) {
                nppiResize_32f_C1R(
                    img_in + b * n_pixels,
                    src_W * sizeof(float),
                    {src_W, src_H},
                    {0, 0, src_W, src_H},
                    img_out + b * n_pixels_dst,
                    dst_W * sizeof(float),
                    {dst_W, dst_H},
                    {0, 0, dst_W, dst_H},
                    bilinear ? NPPI_INTER_LINEAR : NPPI_INTER_NN);
            }

            cudaDeviceSynchronize();
            system_clock::time_point t2 = system_clock::now();
            duration_ms dur = t2 - t1;
            std::cout << "time (ms): " << dur.count() << "\n";
        }
    }

    for (int b = 0; b < B; ++b) {
        cv::Mat img_out_cv(dst_H, dst_W, CV_32FC1);
        cudaMemcpy(
            img_out_cv.data,
            img_out + b * n_pixels_dst,
            n_bytes_dst,
            cudaMemcpyDeviceToHost);
        cv::imwrite(fmt::format("../Data/test{}_post.png", b), img_out_cv);
    }

    return 0;
}
