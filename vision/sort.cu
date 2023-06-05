#include <cub/cub.cuh>

#include <chrono>
#include <iostream>
namespace chrono = std::chrono;
using std::chrono::system_clock;
using std::chrono::time_point;
using duration_ms = chrono::duration<double, std::milli>;

void raidxSortSingle(float *img_in, float *img_out, int B, int H, int W) {
    void *temp_storage_d;
    size_t temp_storage_bytes = 100 * 1024 * 1024;
    cudaMalloc(&temp_storage_d, temp_storage_bytes);

    for (int i = 0; i < 10; ++i) {

        cudaDeviceSynchronize();
        system_clock::time_point t1 = system_clock::now();

        for (int i = 0; i < B; ++i) {
            cub::DeviceRadixSort::SortKeys(
                temp_storage_d,
                temp_storage_bytes,
                img_in + i * W * H,
                img_out + i * W * H,
                W * H);
        }

        cudaDeviceSynchronize();
        system_clock::time_point t2 = system_clock::now();
        duration_ms dur = t2 - t1;
        std::cout << "time (ms): " << dur.count() << "\n";
    }

    cudaFree(temp_storage_d);
}

void raidxSortBatch(float *img_in, float *img_out, int B, int H, int W) {
    void *temp_storage_d;
    size_t temp_storage_bytes = 100 * 1024 * 1024;
    cudaMalloc(&temp_storage_d, temp_storage_bytes);

    int *d_offsets;
    cudaMalloc(&d_offsets, (B + 1) * sizeof(int));

    int h_offsets[B + 1];
    for (int i = 0; i < B + 1; ++i) {
        h_offsets[i] = i * W * H;
    }
    cudaMemcpy(d_offsets, h_offsets, (B + 1) * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < 10; ++i) {

        cudaDeviceSynchronize();
        system_clock::time_point t1 = system_clock::now();

        // DeviceSegmentedRadixSort보다 조금 빠른듯.
        cub::DeviceSegmentedSort::SortKeys(
            temp_storage_d,
            temp_storage_bytes,
            img_in,
            img_out,
            B * H * W,
            B,
            d_offsets,
            d_offsets + 1);

        cudaDeviceSynchronize();
        system_clock::time_point t2 = system_clock::now();
        duration_ms dur = t2 - t1;
        std::cout << "time (ms): " << dur.count() << "\n";
    }

    cudaFree(temp_storage_d);

    cudaFree(d_offsets);
}
