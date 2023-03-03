#include "kernels/test.cuh"
#include <iostream>
#include <vector>

int main() {
    std::vector<float> aaa = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<float> bbb = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<float> ccc(10);

    vecAdd(aaa.data(), bbb.data(), ccc.data(), 10);

    for (const auto &each : ccc) {
        std::cout << each << std::endl;
    }
    return 0;
}
