#include <cuda_runtime.h>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

// CUDA核函数：扁平化数据处理，每个线程生成一个猜测
__global__ void generate_guesses_kernel_flat(const char *flat_values, int value_len, 
                                           const char *d_guess_prefix, int prefix_len, 
                                           char *result_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        char *dst = result_data + idx * (prefix_len + value_len + 1);
        
        // 复制前缀
        if (prefix_len > 0) {
            for (int i = 0; i < prefix_len; ++i) {
                dst[i] = d_guess_prefix[i];
            }
        }
        
        // 复制当前值
        const char *src = flat_values + idx * (value_len + 1);
        for (int i = 0; i < value_len; ++i) {
            dst[prefix_len + i] = src[i];
        }
        dst[prefix_len + value_len] = '\0';
    }
}

// CUDA密码生成接口函数
void cuda_generate_guesses(const char *flat_values, int value_len, 
                          const char *h_guess_prefix, int prefix_len, int n, 
                          std::vector<std::string> &guesses, size_t offset,
                          double* prepare_time, double* kernel_time, double* collect_time)
{
    using namespace std::chrono;
    
    auto t_prepare_start = high_resolution_clock::now();
    
    // 计算内存大小
    size_t flat_size = n * (value_len + 1);
    size_t result_size = n * (prefix_len + value_len + 1);
    
    // 分配设备内存
    char *d_flat_values, *d_guess_prefix = nullptr, *d_result_data;
    
    cudaMalloc(&d_flat_values, flat_size * sizeof(char));
    cudaMemcpy(d_flat_values, flat_values, flat_size * sizeof(char), cudaMemcpyHostToDevice);
    
    if (prefix_len > 0) {
        cudaMalloc(&d_guess_prefix, prefix_len * sizeof(char));
        cudaMemcpy(d_guess_prefix, h_guess_prefix, prefix_len * sizeof(char), cudaMemcpyHostToDevice);
    }
    
    cudaMalloc(&d_result_data, result_size * sizeof(char));
    
    auto t_prepare_end = high_resolution_clock::now();
    if (prepare_time) *prepare_time = duration<double>(t_prepare_end - t_prepare_start).count();
    
    // 执行核函数
    auto t_kernel_start = high_resolution_clock::now();
    
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    generate_guesses_kernel_flat<<<grid_size, block_size>>>(
        d_flat_values, value_len, d_guess_prefix, prefix_len, d_result_data, n
    );
    
    cudaDeviceSynchronize();
    
    auto t_kernel_end = high_resolution_clock::now();
    if (kernel_time) *kernel_time = duration<double>(t_kernel_end - t_kernel_start).count();
    
    // 收集结果
    auto t_collect_start = high_resolution_clock::now();
    
    char *h_result_data = new char[result_size];
    cudaMemcpy(h_result_data, d_result_data, result_size * sizeof(char), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n; ++i) {
        guesses[offset + i].assign(h_result_data + i * (prefix_len + value_len + 1));
    }
    
    auto t_collect_end = high_resolution_clock::now();
    if (collect_time) *collect_time = duration<double>(t_collect_end - t_collect_start).count();
    
    // 释放内存
    delete[] h_result_data;
    cudaFree(d_flat_values);
    cudaFree(d_result_data);
    if (d_guess_prefix) cudaFree(d_guess_prefix);
}