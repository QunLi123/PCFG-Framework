#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace std;
using namespace chrono;
// 全局性能统计变量
double total_prepare_time = 0, total_kernel_time = 0, total_collect_time = 0;

int main()
{
    #ifdef USE_CUDA
    int device_id = 0;
    cudaSetDevice(device_id);
    cout << "CUDA enabled - using GPU acceleration" << endl;
    #else
    cout << "Serial implementation - CPU only" << endl;
    #endif
    
    double time_hash = 0;   // MD5哈希计算时间
    double time_guess = 0;  // 密码生成总时间
    double time_train = 0;  // 模型训练时间
    
    PriorityQueue password_queue;
    
    // 模型训练阶段
    auto train_start = system_clock::now();
    password_queue.m.train("../串行/Rockyou-singleLined-full.txt");
    password_queue.m.order();
    auto train_end = system_clock::now();
    auto train_duration = duration_cast<microseconds>(train_end - train_start);
    time_train = double(train_duration.count()) * microseconds::period::num / microseconds::period::den;
    
    // 初始化优先队列
    password_queue.init();
    cout << "Initialization completed" << endl;
    
    int current_count = 0;
    auto guess_start = system_clock::now();
    
    // 记录历史生成的密码总数
    int history_count = 0;
    
    // 密码生成主循环
    while (!password_queue.priority.empty())
    {
        size_t before_size = password_queue.guesses.size();
        password_queue.PopNext();
        size_t after_size = password_queue.guesses.size();
        password_queue.total_guesses = after_size;
        
        // 累计性能统计
        extern double last_prepare_time, last_kernel_time, last_collect_time;
        total_prepare_time += last_prepare_time;
        total_kernel_time += last_kernel_time;
        total_collect_time += last_collect_time;
        
        // 定期输出进度
        if (password_queue.total_guesses - current_count >= 100000)
        {
            current_count = password_queue.total_guesses;
            
            // 设置生成密码数量上限
            int max_generate_count = 10000000;
            if (history_count + password_queue.total_guesses > max_generate_count)
            {
                auto guess_end = system_clock::now();
                auto guess_duration = duration_cast<microseconds>(guess_end - guess_start);
                time_guess = double(guess_duration.count()) * microseconds::period::num / microseconds::period::den;
                
                cout << "Password generation time: " << time_guess - time_hash << " seconds" << endl;
                cout << "Hash computation time: " << time_hash << " seconds" << endl;
                cout << "Model training time: " << time_train << " seconds" << endl;
                
                #ifdef USE_CUDA
                cout << "CUDA Performance Breakdown:" << endl;
                cout << "  Prepare time: " << total_prepare_time << "s" << endl;
                cout << "  Kernel time: " << total_kernel_time << "s" << endl;
                cout << "  Collect time: " << total_collect_time << "s" << endl;
                #endif
                
                break;
            }
        }
        
        // 内存管理 - 定期清理密码缓存并进行哈希计算
        if (current_count > 1000000)
        {
            auto hash_start = system_clock::now();
            bit32 hash_state[4];
            
            for (const string& password : password_queue.guesses)
            {
                // MD5哈希计算
                MD5Hash(password, hash_state);
                
                // 可选：输出密码和哈希值（用于调试）
                // cout << password << "\t";
                // for (int i = 0; i < 4; i++) {
                //     cout << std::setw(8) << std::setfill('0') << hex << hash_state[i];
                // }
                // cout << endl;
            }
            
            // 计算哈希时间
            auto hash_end = system_clock::now();
            auto hash_duration = duration_cast<microseconds>(hash_end - hash_start);
            time_hash += double(hash_duration.count()) * microseconds::period::num / microseconds::period::den;
            
            // 更新历史计数并清理内存
            history_count += current_count;
            current_count = 0;
            password_queue.guesses.clear();
        }
    }
    
    return 0;
}