#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <format>
#include <random>
#include <ranges>
#include <string>
#include <vector>
#include <mma.h>
#include <cuda_bf16.h>

constexpr int TILE_X = 16;
constexpr int TILE_Y = 16;

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
inline void check_cuda(cudaError_t stat, const char* func, const char* file, const int line) {
    if (stat != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(stat)
            << " in " << file << ":" << line << " -> " << func << "\n";
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

template<typename T, typename U>
bool compare(const std::vector<T>& in1, const std::vector<U>& in2, const double eps) {
    assert(in1.size() == in2.size());
    return std::ranges::equal(in1, in2, [eps](const T a, const U b) {
        return std::fabs(a - b) <= eps;
    });
}

std::vector<__nv_bfloat16> convert_to_half(const std::vector<double>& in) {
    std::vector<__nv_bfloat16> out(in.size());
    std::ranges::transform(in, std::ranges::begin(out), [](const double e) { 
        return __float2bfloat16(static_cast<float>(e));
    });
    return out;
}

__global__ void wmma_gemm(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
    const int A_rows, const int A_cols, const int B_rows, const int B_cols) {
    using namespace nvcuda;

    constexpr int M = 16, N = 16, K = 16;
    int a_row = blockIdx.y * M;
    int b_col = blockIdx.x * N;
    if  (a_row >= A_rows || b_col >= B_cols) {
        return;
    }
    
    wmma::fragment<wmma::matrix_a, M, N, K, __nv_bfloat16, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, __nv_bfloat16, wmma::row_major> B_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> C_frag;
    wmma::fill_fragment(C_frag, 0.f);

    for (int i = 0; i < A_cols; i+=K) {
        wmma::load_matrix_sync(A_frag, &A[a_row * A_cols + i], A_cols);
        wmma::load_matrix_sync(B_frag, &B[i * B_cols + b_col], B_cols);
        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
    }
    wmma::store_matrix_sync(&C[a_row * B_cols + b_col], C_frag, B_cols, wmma::mem_row_major);
}

__global__ void tiled_gemm(const double* A, const double* B, double* C,
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

__global__ void gemm(const double* A, const double* B, double* C,
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
    std::uniform_real_distribution<double> urd(1.0, 16.0);

    std::vector<double> A(A_rows * A_cols, 0.0);
    std::vector<double> B(B_rows * B_cols, 0.0);
    std::vector<double> C_ref(A_rows * B_cols, 0.0);
    std::vector<double> C_gemm(A_rows * B_cols, 0.0);
    std::vector<double> C_tiled_gemm(A_rows * B_cols, 0.0);

    std::ranges::generate(A, [&]() { return urd(rng); });
    std::ranges::generate(B, [&]() { return urd(rng); });
    
    const auto begin = std::chrono::high_resolution_clock::now();
    cpu_gemm(A, B, C_ref, A_rows, A_cols, B_rows, B_cols);
    const auto end = std::chrono::high_resolution_clock::now();
    std::cout << "CPU gemm: " << std::chrono::duration<double>(end - begin) << " s\n";

    double* d_a;
    double* d_b;
    double* d_c;

    checkCudaErrors(cudaMalloc(&d_a, A.size() * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_b, B.size() * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_c, C_ref.size() * sizeof(double)));

    checkCudaErrors(cudaMemcpy(d_a, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice));
    
    dim3 blockDim{TILE_X, TILE_Y};
    dim3 gridDim{(B_cols + blockDim.x - 1) / blockDim.x, (A_rows + blockDim.y - 1) / blockDim.y};

    float gemm_time = 0, tile_time = 0, wmma_time = 0;
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start));
    gemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, A_rows, A_cols, B_rows, B_cols);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&gemm_time, start, stop));
    std::cout << "CUDA gemm: " << gemm_time / 1000.0f << " s\n";

    checkCudaErrors(cudaMemcpy(C_gemm.data(), d_c, C_gemm.size() * sizeof(double), cudaMemcpyDeviceToHost));
    assert((compare<double, double>(C_ref, C_gemm, 1e-6)));


    cudaMemset(d_c, 0.0, A_rows * B_cols);

    checkCudaErrors(cudaEventRecord(start));
    tiled_gemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, A_rows, A_cols, B_rows, B_cols);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&tile_time, start, stop));
    std::cout << "CUDA tiled_gemm: " << tile_time / 1000.0f << " s\n";
    
    checkCudaErrors(cudaMemcpy(C_tiled_gemm.data(), d_c, C_tiled_gemm.size() * sizeof(double), cudaMemcpyDeviceToHost));
    assert((compare<double, double>(C_ref, C_tiled_gemm, 1e-6)));


    __nv_bfloat16* d_a_wmma;
    __nv_bfloat16* d_b_wmma;
    float* d_c_wmma;

    std::vector<__nv_bfloat16> A_half = std::move(convert_to_half(A));
    std::vector<__nv_bfloat16> B_half = std::move(convert_to_half(B));
    std::vector<float> C_wmma_gemm(A_rows * B_cols, 0.f);
    
    checkCudaErrors(cudaMalloc(&d_a_wmma, A_half.size() * sizeof(__nv_bfloat16)));
    checkCudaErrors(cudaMalloc(&d_b_wmma, B_half.size() * sizeof(__nv_bfloat16)));
    checkCudaErrors(cudaMalloc(&d_c_wmma, C_wmma_gemm.size() * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_a_wmma, A_half.data(), A_half.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b_wmma, B_half.data(), B_half.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaEventRecord(start));
    wmma_gemm<<<gridDim, blockDim>>>(d_a_wmma, d_b_wmma, d_c_wmma, A_rows, A_cols, B_rows, B_cols);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&wmma_time, start, stop));
    std::cout << "CUDA wmma_gemm: " << wmma_time / 1000.0f << " s\n";

    checkCudaErrors(cudaMemcpy(C_wmma_gemm.data(), d_c_wmma, C_wmma_gemm.size() * sizeof(float), cudaMemcpyDeviceToHost));


    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));
    checkCudaErrors(cudaFree(d_a_wmma));
    checkCudaErrors(cudaFree(d_b_wmma));
    checkCudaErrors(cudaFree(d_c_wmma));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    return 0;
}
