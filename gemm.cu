#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <format>
#include <random>
#include <ranges>
#include <string>
#include <vector>

#define TILE_X 32
#define TILE_Y 32

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
inline void check_cuda(cudaError_t code, const char* func, const char* file, const int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error " << code << ": " << cudaGetErrorString(code) << " in "
            << file << ":" << line << " -> " << func << "\n";
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

bool compare(const std::vector<double>& in1, const std::vector<double>& in2, double epsilon=1e-6) {
    assert(in1.size() == in2.size());
    return std::ranges::equal(in1, in2, [epsilon](const double a, const double b) {
        return std::fabs(a - b) <= epsilon;
    });
}

__global__ void tiled_gemm(double* A, double* B, double* C,
    const int A_rows, const int A_cols, const int B_rows, const int B_cols) {
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int row = TILE_Y * blockIdx.y + ty;
    const int col = TILE_X * blockIdx.x + tx;

    __shared__ double s_a[TILE_Y][TILE_X];
    __shared__ double s_b[TILE_Y][TILE_X];
    
    double dot_prod = 0.0;
    for (int tile_offset = 0; tile_offset < A_cols; tile_offset+=TILE_X) {
        s_a[ty][tx] = (tile_offset + tx < A_cols && row < A_rows) ? A[row * A_cols + tile_offset + tx] : 0.0;
        s_b[ty][tx] = (tile_offset + ty < A_cols && col < B_cols) ? B[(tile_offset + ty) * B_cols + col] : 0.0;
        __syncthreads();

        for (int i = 0; i < TILE_X; i++) {
            dot_prod += s_a[ty][i] * s_b[i][tx];
        }
        __syncthreads();
    }
    if (row < A_rows && col < B_cols) {
        C[row * B_cols + col] = dot_prod;
    }
}

__global__ void gemm(double* A, double* B, double* C,
    const int A_rows, const int A_cols, const int B_rows, const int B_cols) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows || col >= B_cols) {
        return;
    }
    double dot_prod = 0.0;
    for (int i = 0; i < A_cols; i++) {
        dot_prod += A[row * A_cols + i] * B[i * B_cols + col];
    }
    C[row * B_cols + col] = dot_prod;
}

void cpu_gemm(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C,
    const int A_rows, const int A_cols, const int B_rows, const int B_cols) {
    for (int i = 0; i < A_rows; i++) {   
        for (int k = 0; k < A_cols; k++) {
            const double r = A[i * A_cols + k];
            for (int j = 0; j < B_cols; j++) {
                C[i * B_cols + j] += r * B[k * B_cols + j];
            }
        }
    }
}

int main(int argc, char** argv) {
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
    std::vector<double> C_tiled_gemm(A_rows * B_cols, 0.0);

    std::ranges::generate(A.begin(), A.end(), [&](){ return urd(rng); });
    std::ranges::generate(B.begin(), B.end(), [&](){ return urd(rng); });
    
    cpu_gemm(A, B, C_ref, A_rows, A_cols, B_rows, B_cols);

    double* d_a;
    double* d_b;
    double* d_c;

    checkCudaErrors(cudaMalloc(&d_a, A.size() * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_b, B.size() * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_c, C_ref.size() * sizeof(double)));

    checkCudaErrors(cudaMemcpy(d_a, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice));
    
    dim3 TPB{32, 32};
    dim3 BPG{(B_cols + TPB.x - 1) / TPB.x, (A_rows + TPB.y - 1) / TPB.y};

    float elapsed_time = 0;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start));
    gemm<<<BPG, TPB>>>(d_a, d_b, d_c, A_rows, A_cols, B_rows, B_cols);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start, stop));
    std::cout << "CUDA gemm: " << elapsed_time / 1000.0f << " seconds\n";

    checkCudaErrors(cudaMemcpy(C_gemm.data(), d_c, C_gemm.size() * sizeof(double), cudaMemcpyDeviceToHost));
    assert(compare(C_ref, C_gemm));

    cudaMemset(d_c, 0.0, A_rows * B_cols);

    checkCudaErrors(cudaEventRecord(start));
    tiled_gemm<<<BPG, TPB>>>(d_a, d_b, d_c, A_rows, A_cols, B_rows, B_cols);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsed_time, start, stop));
    std::cout << "CUDA tiled_gemm: " << elapsed_time / 1000.0f << " seconds\n";
    
    checkCudaErrors(cudaMemcpy(C_tiled_gemm.data(), d_c, C_tiled_gemm.size() * sizeof(double), cudaMemcpyDeviceToHost));
    assert(compare(C_ref, C_tiled_gemm));

    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    return 0;
}
