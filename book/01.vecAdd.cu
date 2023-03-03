#include <vector>
#include <iostream>

__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}


void vecAdd(float* A, float* B, float* C, int n) {

    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);
    cudaMalloc((void**) &C_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    int N = 100000;
    std::vector<float> aaa(N);
    std::vector<float> bbb(N);
    std::vector<float> ccc(N);
    for(int i=0; i<N; ++i) {
        aaa[i] = i;
        bbb[i] = i;
    }

    vecAdd(aaa.data(), bbb.data(), ccc.data(), N);

    for(int i=0; i<N; ++i) {
        if (aaa[i] + bbb[i] != ccc[i]) {
            std::cout << "fail\n";
            return 0;
        }
    }
    std::cout << "success\n";
    return 0;
}
