#include <opencv2/cudafilters.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cvcuda/OpAverageBlur.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>

#include <nppi.h>

#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
namespace chrono = std::chrono;
using std::chrono::system_clock;
using duration_ms = chrono::duration<double, std::milli>;

// clang-format off
void boxFilterLarge(float *d_src, float *d_temp, float *d_dest, int B, int H, int W, int radius, int nthreads);
void boxFilterSmall(float *d_src, float *d_dest, int B, int H, int W, int radius);
void tiledBoxFilter(float *d_src, float *d_dest, int B, int H, int W, int radius);
// clang-format on

int main() {

    int B = 8;
    int H = 1080;
    int W = 1920;
    int radius = 5;

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
        std::cout << "\nBoxFilterLarge: \n";
        float *img_temp;
        cudaMalloc(&img_temp, B * n_bytes);
        for (int i = 0; i < 10; ++i) {

            cudaDeviceSynchronize();
            system_clock::time_point t1 = system_clock::now();

            boxFilterLarge(img_in, img_temp, img_out, B, H, W, radius, 16);

            cudaDeviceSynchronize();
            system_clock::time_point t2 = system_clock::now();
            duration_ms dur = t2 - t1;
            std::cout << "time (ms): " << dur.count() << "\n";
        }
    }

    if (true) {
        std::cout << "\nBoxFilterSmall: \n";
        float *img_temp;
        cudaMalloc(&img_temp, B * n_bytes);
        for (int i = 0; i < 10; ++i) {

            cudaDeviceSynchronize();
            system_clock::time_point t1 = system_clock::now();

            boxFilterSmall(img_in, img_out, B, H, W, radius);

            cudaDeviceSynchronize();
            system_clock::time_point t2 = system_clock::now();
            duration_ms dur = t2 - t1;
            std::cout << "time (ms): " << dur.count() << "\n";
        }
    }

    if (true) {
        std::cout << "\nTiledBoxFilter: \n";
        for (int i = 0; i < 10; ++i) {

            cudaDeviceSynchronize();
            system_clock::time_point t1 = system_clock::now();

            tiledBoxFilter(img_in, img_out, B, H, W, radius);

            cudaDeviceSynchronize();
            system_clock::time_point t2 = system_clock::now();
            duration_ms dur = t2 - t1;
            std::cout << "time (ms): " << dur.count() << "\n";
        }
    }

    if (true) {
        std::cout << "\nOpenCV CUDA: \n";

        auto filter =
            cv::cuda::createBoxFilter(CV_32FC1, CV_32FC1, {2 * radius + 1, 2 * radius + 1});

        for (int i = 0; i < 10; ++i) {
            cudaDeviceSynchronize();
            system_clock::time_point t1 = system_clock::now();

            for (int b = 0; b < B; ++b) {
                cv::cuda::GpuMat in{H, W, CV_32FC1, img_in + b * n_pixels};
                cv::cuda::GpuMat out{H, W, CV_32FC1, img_out + b * n_pixels};
                filter->apply(in, out);
            }

            cudaDeviceSynchronize();
            system_clock::time_point t2 = system_clock::now();
            duration_ms dur = t2 - t1;
            std::cout << "time (ms): " << dur.count() << "\n";
        }
    }

    if (true) {
        std::cout << "\nNPPI: \n";

        uint8_t *p_buffer;
        cudaMalloc(&p_buffer, 30 * 1024 * 1024);

        NppStreamContext nppStreamContext{};
        for (int i = 0; i < 10; ++i) {
            cudaDeviceSynchronize();
            system_clock::time_point t1 = system_clock::now();

            for (int b = 0; b < 8; ++b) {
                nppiFilterBoxBorderAdvanced_32f_C1R_Ctx(
                    img_in + b * n_pixels,
                    W * sizeof(float),
                    {W, H},
                    {0, 0},
                    img_out + b * n_pixels,
                    W * sizeof(float),
                    {W, H},
                    {2 * radius + 1, 2 * radius + 1},
                    {radius, radius},
                    NPP_BORDER_REPLICATE,
                    p_buffer,
                    nppStreamContext);
            }

            cudaDeviceSynchronize();
            system_clock::time_point t2 = system_clock::now();
            duration_ms dur = t2 - t1;
            std::cout << "time (ms): " << dur.count() << "\n";
        }

        cudaFree(p_buffer);
    }

    if (true) {
        std::cout << "\nCV-CUDA: \n";
        nvcv::TensorDataStridedCuda::Buffer inBuf;
        inBuf.strides[3] = sizeof(float);
        inBuf.strides[2] = 1 * inBuf.strides[3];
        inBuf.strides[1] = W * inBuf.strides[2];
        inBuf.strides[0] = H * inBuf.strides[1];
        inBuf.basePtr = (NVCVByte *)img_in;

        nvcv::Tensor::Requirements inReqs =
            nvcv::Tensor::CalcRequirements(B, {W, H}, nvcv::FMT_F32);

        nvcv::TensorDataStridedCuda inData(
            nvcv::TensorShape{inReqs.shape, inReqs.rank, inReqs.layout},
            nvcv::DataType{inReqs.dtype},
            inBuf);

        nvcv::TensorWrapData inTensor(inData);
        ///
        ///
        nvcv::TensorDataStridedCuda::Buffer inBuf2;
        inBuf2.strides[3] = sizeof(float);
        inBuf2.strides[2] = 1 * inBuf2.strides[3];
        inBuf2.strides[1] = W * inBuf2.strides[2];
        inBuf2.strides[0] = H * inBuf2.strides[1];
        inBuf2.basePtr = (NVCVByte *)img_out;

        nvcv::Tensor::Requirements inReqs2 =
            nvcv::Tensor::CalcRequirements(B, {W, H}, nvcv::FMT_F32);

        nvcv::TensorDataStridedCuda inData2(
            nvcv::TensorShape{inReqs2.shape, inReqs2.rank, inReqs2.layout},
            nvcv::DataType{inReqs2.dtype},
            inBuf2);

        nvcv::TensorWrapData outTensor(inData2);
        ///

        cvcuda::AverageBlur averageBlurOp({2 * radius + 1, 2 * radius + 1}, B);

        nvcv::Size2D kernelSize(2 * radius + 1, 2 * radius + 1);
        int2 kernelAnchor{-1, -1};

        for (int i = 0; i < 10; ++i) {
            cudaDeviceSynchronize();
            system_clock::time_point t1 = system_clock::now();

            averageBlurOp(
                0,
                inTensor,
                outTensor,
                kernelSize,
                kernelAnchor,
                NVCV_BORDER_REFLECT101);

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
