#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
using namespace std;
using namespace chrono;

// 全局变量池化 - 在程序启动时预分配，避免反复创建销毁
namespace GlobalBuffers {
    // 主进程使用的缓冲区
    vector<vector<PT>> worker_batches;
    vector<char> receive_buffer;
    
    // 工作进程使用的缓冲区
    vector<char> serialize_buffer;
    vector<char> send_buffer;
    vector<string> all_guesses;
    
    // 初始化函数
    void init_master_buffers(int num_workers) {
        worker_batches.resize(num_workers);
        for (auto& batch : worker_batches) {
            batch.reserve(20);  // 预分配batch容量
        }
        receive_buffer.reserve(200000 * 64);  // 预分配接收缓冲区
    }
    
    void init_worker_buffers() {
        serialize_buffer.reserve(10000);  // 预分配序列化缓冲区
        send_buffer.reserve(100000 * 64); // 预分配发送缓冲区
        all_guesses.reserve(50000);       // 预分配密码容器
    }
    
    void clear_master_batch() {
        for (auto& batch : worker_batches) {
            batch.clear();  // 清空但保持容量
        }
    }
    
    void clear_worker_data() {
        all_guesses.clear();  // 清空但保持容量
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time_hash = 0, time_guess = 0, time_train = 0;
    PriorityQueue q;
    int batch_size = 8;  // 使用你当前的batch_size

    if (rank == 0)
    {
        // 初始化主进程缓冲区
        GlobalBuffers::init_master_buffers(size - 1);
        
        auto start_train = system_clock::now();
        q.m.train("../Rockyou-singleLined-full.txt");
        q.m.order();
        auto end_train = system_clock::now();
        auto duration_train = duration_cast<microseconds>(end_train - start_train);
        time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

        q.init();
        cout << "Initial PT count: " << q.priority.size() << endl;

        int curr_num = 0, total_generated = 0;
        int generate_limit = 10000000;
        auto start = system_clock::now();

        while (total_generated < generate_limit)
        {
            int k = size - 1;
            
            // 使用全局缓冲区，避免重复创建
            GlobalBuffers::clear_master_batch();
            int pt_idx = 0;

            // 分发PT到全局缓冲区
            while (pt_idx < k * batch_size && !q.priority.empty())
            {
                GlobalBuffers::worker_batches[pt_idx % k].emplace_back(std::move(q.priority.front()));
                q.priority.erase(q.priority.begin());
                pt_idx++;
            }

            // 发送PT给worker
            for (int i = 0; i < k; ++i)
            {
                for (int j = 0; j < batch_size; ++j)
                {
                    if (j < GlobalBuffers::worker_batches[i].size()) {
                        auto pt_str = serialize_PT(GlobalBuffers::worker_batches[i][j]);
                        int len = pt_str.size();
                        MPI_Send(&len, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
                        MPI_Send(pt_str.data(), len, MPI_CHAR, i + 1, 0, MPI_COMM_WORLD);
                    }
                }
            }

            // 立即生成新PT
            for (auto &batch : GlobalBuffers::worker_batches)
            {
                for (auto &processed_pt : batch)
                {
                    vector<PT> new_pts = processed_pt.NewPTs();  // 这个暂不池化，大小不确定
                    for (const PT &new_pt : new_pts)
                    {
                        PT pt_copy = new_pt;
                        q.CalProb(pt_copy);

                        // 按概率插入
                        bool inserted = false;
                        for (auto iter = q.priority.begin(); iter != q.priority.end(); ++iter)
                        {
                            if (pt_copy.prob > iter->prob)
                            {
                                q.priority.insert(iter, pt_copy);
                                inserted = true;
                                break;
                            }
                        }
                        if (!inserted)
                        {
                            q.priority.push_back(pt_copy);
                        }
                    }
                }
            }

            // 接收worker结果
            for (int i = 0; i < k; ++i)
            {
                int total_guess_count = 0;
                MPI_Recv(&total_guess_count, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // 使用全局接收缓冲区，确保容量足够
                size_t required_size = total_guess_count * 64;
                if (GlobalBuffers::receive_buffer.capacity() < required_size) {
                    GlobalBuffers::receive_buffer.reserve(required_size * 2);  // 预留更多空间
                }
                GlobalBuffers::receive_buffer.resize(required_size);
                
                MPI_Recv(GlobalBuffers::receive_buffer.data(), required_size, MPI_CHAR, 
                        i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                q.guesses.reserve(q.guesses.size() + total_guess_count);
                for (int j = 0; j < total_guess_count; ++j)
                {
                    q.guesses.emplace_back(GlobalBuffers::receive_buffer.data() + j * 64, 
                                          strnlen(GlobalBuffers::receive_buffer.data() + j * 64, 64));
                }
                curr_num += total_guess_count;
                total_generated += total_guess_count;
            }

            if (curr_num > 1000000)
            {
                auto start_hash = system_clock::now();
                bit32 state[4];
                for (const string &pw : q.guesses)
                {
                    MD5Hash(pw, state);
                }
                auto end_hash = system_clock::now();
                auto duration = duration_cast<microseconds>(end_hash - start_hash);
                time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

                cout << "Generated: " << total_generated << ", Queue size: " << q.priority.size() << endl;
                curr_num = 0;
                q.guesses.clear();
            }
        }

        // 停止所有worker
        for (int i = 1; i < size; ++i)
        {
            int terminate_len = -1;
            MPI_Send(&terminate_len, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        auto end = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;

        cout << "Total generated: " << total_generated << endl;
        cout << "Train time: " << time_train << " seconds" << endl;
        cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
        cout << "Hash time: " << time_hash << " seconds" << endl;
    }
    else
    {
        // 初始化工作进程缓冲区
        GlobalBuffers::init_worker_buffers();
        
        // worker进程：每个进程独立初始化模型
        q.m.train("../Rockyou-singleLined-full.txt");
        q.m.order();
        q.init();
        
        while (true)
        {
            // 使用全局缓冲区，避免重复创建
            GlobalBuffers::clear_worker_data();
            bool should_terminate = false;
            
            for (int b = 0; b < batch_size; ++b)
            {
                int len = 0;
                MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (len == -1)
                {
                    should_terminate = true;
                    break;
                }
                
                // 使用全局序列化缓冲区
                if (GlobalBuffers::serialize_buffer.capacity() < len) {
                    GlobalBuffers::serialize_buffer.reserve(len * 2);
                }
                GlobalBuffers::serialize_buffer.resize(len);
                
                MPI_Recv(GlobalBuffers::serialize_buffer.data(), len, MPI_CHAR, 
                        0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                PT pt = deserialize_PT(GlobalBuffers::serialize_buffer);

                // 直接生成到全局密码容器
                q.Generate(pt, GlobalBuffers::all_guesses);
            }
            
            if (should_terminate)
            {
                break;
            }

            // 发送结果
            int total_guess_count = GlobalBuffers::all_guesses.size();
            MPI_Send(&total_guess_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            
            // 使用全局发送缓冲区
            size_t required_size = total_guess_count * 64;
            if (GlobalBuffers::send_buffer.capacity() < required_size) {
                GlobalBuffers::send_buffer.reserve(required_size * 2);
            }
            GlobalBuffers::send_buffer.resize(required_size);
            std::fill(GlobalBuffers::send_buffer.begin(), GlobalBuffers::send_buffer.end(), 0);
            
            for (int i = 0; i < total_guess_count; ++i)
            {
                strncpy(GlobalBuffers::send_buffer.data() + i * 64, 
                       GlobalBuffers::all_guesses[i].c_str(), 63);
            }
            MPI_Send(GlobalBuffers::send_buffer.data(), required_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }
    
    MPI_Finalize();
    return 0;
}

// 序列化函数保持不变，但可以考虑使用全局缓冲区优化
std::vector<char> serialize_PT(const PT &pt)
{
    std::vector<char> buffer;
    size_t estimated_size = sizeof(int) * (5 + pt.content.size() * 2 +
                                           pt.curr_indices.size() + pt.max_indices.size());
    buffer.reserve(estimated_size);

    auto append_int = [&buffer](int value)
    {
        const char *bytes = reinterpret_cast<const char *>(&value);
        buffer.insert(buffer.end(), bytes, bytes + sizeof(int));
    };

    append_int(static_cast<int>(pt.content.size()));
    for (const auto &seg : pt.content)
    {
        append_int(seg.type);
        append_int(seg.length);
    }
    append_int(pt.pivot);
    append_int(static_cast<int>(pt.curr_indices.size()));
    for (int idx : pt.curr_indices)
    {
        append_int(idx);
    }
    append_int(static_cast<int>(pt.max_indices.size()));
    for (int idx : pt.max_indices)
    {
        append_int(idx);
    }

    return buffer;
}

PT deserialize_PT(const std::vector<char> &buffer)
{
    PT pt;
    size_t offset = 0;

    auto read_int = [&buffer, &offset]() -> int
    {
        if (offset + sizeof(int) > buffer.size())
        {
            throw std::runtime_error("Buffer underflow");
        }
        int value = *reinterpret_cast<const int *>(buffer.data() + offset);
        offset += sizeof(int);
        return value;
    };

    int content_size = read_int();
    pt.content.reserve(content_size);
    for (int i = 0; i < content_size; ++i)
    {
        int type = read_int();
        int length = read_int();
        pt.content.emplace_back(type, length);
    }

    pt.pivot = read_int();
    int curr_size = read_int();
    pt.curr_indices.reserve(curr_size);
    for (int i = 0; i < curr_size; ++i)
    {
        pt.curr_indices.emplace_back(read_int());
    }

    int max_size = read_int();
    pt.max_indices.reserve(max_size);
    for (int i = 0; i < max_size; ++i)
    {
        pt.max_indices.emplace_back(read_int());
    }

    return pt;
}