#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <iostream>
#include <vector>

__global__ void colorToGray(uint8_t *Pin, uint8_t *Pout, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int gray_offset = row * width + col;
        int rgb_offset = gray_offset * 3;

        uint8_t r = Pin[rgb_offset];
        uint8_t g = Pin[rgb_offset + 1];
        uint8_t b = Pin[rgb_offset + 2];

        Pout[gray_offset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

__global__ void blurKernel(uint8_t *in, uint8_t *out, int W, int H) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    constexpr int BLUR_SIZE = 1;

    if (col < W && row < H) {
        int pix_val = 0;
        int pixels = 0;

        // Get average of the surrounding BLUR_SIZE x BLUR_SIZE box
        for (int blur_row = -BLUR_SIZE; blur_row < BLUR_SIZE + 1; ++blur_row) {
            for (int blur_col = -BLUR_SIZE; blur_col < BLUR_SIZE + 1; ++blur_col) {
                int cur_row = row + blur_row;
                int cur_col = col + blur_col;
                if (cur_row >= 0 && cur_row < H && cur_col >= 0 && cur_col < W) {
                    pix_val += in[cur_row * W + cur_col];
                    ++pixels; // Keep track of number of pixels in the avg
                }
            }
        }
        out[row * W + col] = (uint8_t)(pix_val / pixels);
    }
}

int main() {

    cv::Mat rgb_img = cv::imread("../lena_std.tif");
    cv::cvtColor(rgb_img, rgb_img, cv::COLOR_BGR2RGB);
    const int W = rgb_img.cols;
    const int H = rgb_img.rows;

    cv::Mat gray_img(H, W, CV_8UC1);
    cv::Mat gray_blur_img(H, W, CV_8UC1);

    int rgb_img_size = rgb_img.total() * rgb_img.elemSize();
    int gray_img_size = gray_img.total() * gray_img.elemSize();

    uint8_t *rgb_img_d;
    uint8_t *gray_img_d;
    uint8_t *gray_blur_img_d;
    cudaMalloc((void **)&rgb_img_d, rgb_img_size);
    cudaMalloc((void **)&gray_img_d, gray_img_size);
    cudaMalloc((void **)&gray_blur_img_d, gray_img_size);

    cudaMemcpy(rgb_img_d, rgb_img.data, rgb_img_size, cudaMemcpyHostToDevice);

    {
        dim3 dim_grid(ceil(W / 16.0), ceil(H / 16.0), 1);
        dim3 dim_block(16, 16, 1);
        colorToGray<<<dim_grid, dim_block>>>(rgb_img_d, gray_img_d, W, H);

        cudaMemcpy(gray_img.data, gray_img_d, gray_img_size, cudaMemcpyDeviceToHost);
        cv::imwrite("gray.jpg", gray_img);
    }
    {
        dim3 dim_grid(ceil(W / 16.0), ceil(H / 16.0), 1);
        dim3 dim_block(16, 16, 1);
        blurKernel<<<dim_grid, dim_block>>>(gray_img_d, gray_blur_img_d, W, H);

        cudaMemcpy(gray_blur_img.data, gray_blur_img_d, gray_img_size, cudaMemcpyDeviceToHost);
        cv::imwrite("gray_blur.jpg", gray_blur_img);
    }

    cudaFree(rgb_img_d);
    cudaFree(gray_img_d);
    cudaFree(gray_blur_img_d);

    return 0;
}
