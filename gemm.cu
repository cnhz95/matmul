#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <format>
#include <random>
#include <ranges>
#include <string>
#include <vector>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t code, const char* func, const char* file, const int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error " << code << ": " << cudaGetErrorString(code) << " in "
            << file << ":" << line << " -> " << func << "\n";
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

bool array_equals(const std::vector<double>& in1, const std::vector<double>& in2, double epsilon=1e-6) {
    assert(in1.size() == in2.size());
    return std::ranges::equal(in1, in2, [epsilon](const double a, const double b) {
        return std::fabs(a - b) <= epsilon;
    });
}

void gemm_cpu(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C,
    const int A_rows, const int A_cols, const int B_rows, const int B_cols) {
    for (int i = 0; i < A_rows; i++) {   
        for (int k = 0; k < A_cols; k++) {
            double r = A[i * A_cols + k];
            for (int j = 0; j < B_cols; j++) {
                C[i * B_cols + j] += r * B[k * B_cols + j];
            }
        }
    }
}

__global__ void gemm(double* A, double* B, double* C_gemm,
    const int A_rows, const int A_cols, const int B_rows, const int B_cols) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows || col >= B_cols) {
        return;
    }
    double sum = 0.0;
    for (int i = 0; i < A_cols; i++) {
        sum += A[A_cols * row + i] * B[i * B_cols + col];
    }
    C_gemm[row * B_cols + col] = sum;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <A_rows> <A_cols> <B_rows> <B_cols>\n";
        return EXIT_FAILURE;
    }

    const int A_rows = std::stoi(argv[1]);
    const int A_cols = std::stoi(argv[2]);
    const int B_rows = std::stoi(argv[3]);
    const int B_cols = std::stoi(argv[4]);

    assert(A_cols == B_rows);
    std::cout << std::format("A: {}x{}\nB: {}x{}\n", A_rows, A_cols, B_rows, B_cols);

    std::mt19937 rng((uint32_t)std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> urd(1.0, 256.0);

    std::vector<double> A(A_rows * A_cols, 0.0);
    std::vector<double> B(B_rows * B_cols, 0.0);
    std::vector<double> C_ref(A_rows * B_cols, 0.0);
    std::vector<double> C_gemm(A_rows * B_cols, 0.0);

    std::generate(A.begin(), A.end(), [&](){ return urd(rng); });
    std::generate(B.begin(), B.end(), [&](){ return urd(rng); });
    
    gemm_cpu(A, B, C_ref, A_rows, A_cols, B_rows, B_cols);

    double* d_a;
    double* d_b;
    double* d_c;

    checkCudaErrors(cudaMalloc(&d_a, A.size() * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_b, B.size() * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_c, C_gemm.size() * sizeof(double)));

    checkCudaErrors(cudaMemcpy(d_a, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice));

    float elapsed_time = 0;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    
    dim3 TPB = { 32, 32 };
    dim3 BPG = { (B_cols + TPB.x - 1) / TPB.x, (A_rows + TPB.y - 1) / TPB.y };

    checkCudaErrors(cudaEventRecord(start));
    gemm<<<BPG, TPB>>>(d_a, d_b, d_c, A_rows, A_cols, B_rows, B_cols);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start, stop));
    std::cout << "Execution time: " << elapsed_time / 1000.0f << " seconds\n";

    checkCudaErrors(cudaMemcpy(C_gemm.data(), d_c, C_gemm.size() * sizeof(double), cudaMemcpyDeviceToHost));
    assert(array_equals(C_gemm, C_ref));

    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    return 0;
}
