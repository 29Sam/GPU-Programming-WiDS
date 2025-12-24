#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
__global__ void multiplyScale(const float* A, const float* B, float* C, float alpha, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = alpha * A[i] * B[i];
    }
}
int main() {
    int N = 1024;
    float alpha = 2.0f;

    std::vector<float> h_A(N), h_B(N), h_C(N);

    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 0.5f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    multiplyScale<<<blocks, threads>>>(d_A, d_B, d_C, alpha, N);

    cudaMemcpy(h_C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        float expected = alpha * h_A[i] * h_B[i];
        if (std::fabs(h_C[i] - expected) > 1e-5) {
            std::cout << "FAIL at index " << i << std::endl;
            return 1;
        }
    }
    std::cout << "Multiply-and-Scale: PASS" << std::endl;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
