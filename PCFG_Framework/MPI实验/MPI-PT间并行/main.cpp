#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h> // 引入MPI头文件，实现多进程并行
using namespace std;
using namespace chrono;

// 这是pt级多进程代码

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time_hash = 0, time_guess = 0, time_train = 0;
    PriorityQueue q;
    int batch_size = 10;

    if (rank == 0)
    {
        auto start_train = system_clock::now();
        q.m.train("./原始版本/Rockyou-singleLined-full.txt");
        q.m.order();
        auto end_train = system_clock::now();
        auto duration_train = duration_cast<microseconds>(end_train - start_train);
        time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

        q.init();
        cout << "Initial PT count: " << q.priority.size() << endl;

        int curr_num = 0, total_generated = 0;
        int generate_limit = 10000000; // 终止条件
        auto start = system_clock::now();
        

        while (total_generated < generate_limit)
        {
            int k = size - 1;
            vector<vector<PT>> worker_batches(k);
            int pt_idx = 0;

            // 分发PT
            while (pt_idx < k * batch_size)
            {
                worker_batches[pt_idx % k].emplace_back(std::move(q.priority.front()));
                q.priority.erase(q.priority.begin());
                pt_idx++;
            }

            // 发送PT给worker
            for (int i = 0; i < k; ++i)
            {
                // int batch_cnt = worker_batches[i].size();
                // MPI_Send(&batch_cnt, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
                for (int j = 0; j <  batch_size; ++j)
                {
                    auto pt_str = serialize_PT(worker_batches[i][j]);
                    int len = pt_str.size();
                    MPI_Send(&len, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
                    MPI_Send(pt_str.data(), len, MPI_CHAR, i + 1, 0, MPI_COMM_WORLD);
                }
            }

            // 立即生成新PT
            for (auto &batch : worker_batches)
            {
                for (auto &processed_pt : batch)
                {
                    vector<PT> new_pts = processed_pt.NewPTs();
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
                vector<char> buf(total_guess_count * 64);
                MPI_Recv(buf.data(), total_guess_count * 64, MPI_CHAR, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                q.guesses.reserve(q.guesses.size() + total_guess_count); // 预分配空间
                for (int j = 0; j < total_guess_count; ++j)
                {
                    q.guesses.emplace_back(buf.data() + j * 64, strnlen(buf.data() + j * 64, 64));
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
        // worker进程：每个进程独立初始化模型，循环接收PT batch并生成guesses
        q.m.train("./原始版本/Rockyou-singleLined-full.txt");
        q.m.order();
        q.init();
        std::vector<std::string> all_guesses;
        all_guesses.reserve(10000);
        while (true)
        {
            // int batch_cnt = 0;
            // MPI_Recv(&batch_cnt, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // if (batch_cnt == -1)
            //     break;

            all_guesses.clear();
            bool should_terminate = false;
            for (int b = 0; b < batch_size; ++b)
            {
                int len = 0;
                MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (len == -1)
                {
                    // 收到终止信号
                    should_terminate = true;
                    break;
                }
                std::vector<char> buf(len);
                MPI_Recv(buf.data(), len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                PT pt = deserialize_PT(buf);

                // 直接生成到all_guesses，避免中间vector和拷贝
                q.Generate(pt, all_guesses);
            }
            if (should_terminate)
            {
                break;
            }

            int total_guess_count = all_guesses.size();
            MPI_Send(&total_guess_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            std::vector<char> outbuf(total_guess_count * 64, 0);
            for (int i = 0; i < total_guess_count; ++i)
            {
                strncpy(outbuf.data() + i * 64, all_guesses[i].c_str(), 63);
            }
            MPI_Send(outbuf.data(), total_guess_count * 64, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Finalize(); // 结束MPI环境
    return 0;
}

// PT序列化
std::vector<char> serialize_PT(const PT &pt)
{
    std::vector<char> buffer;

    // 预估缓冲区大小，避免频繁resize
    size_t estimated_size = sizeof(int) * (5 + pt.content.size() * 2 +
                                           pt.curr_indices.size() + pt.max_indices.size());
    buffer.reserve(estimated_size);

    // 辅助函数：写入整数
    auto append_int = [&buffer](int value)
    {
        const char *bytes = reinterpret_cast<const char *>(&value);
        buffer.insert(buffer.end(), bytes, bytes + sizeof(int));
    };

    // 写入content大小
    append_int(static_cast<int>(pt.content.size()));

    // 写入content数据
    for (const auto &seg : pt.content)
    {
        append_int(seg.type);
        append_int(seg.length);
    }

    // 写入pivot
    append_int(pt.pivot);

    // 写入curr_indices
    append_int(static_cast<int>(pt.curr_indices.size()));
    for (int idx : pt.curr_indices)
    {
        append_int(idx);
    }

    // 写入max_indices
    append_int(static_cast<int>(pt.max_indices.size()));
    for (int idx : pt.max_indices)
    {
        append_int(idx);
    }

    return buffer;
}

// 反序列化PT
PT deserialize_PT(const std::vector<char> &buffer)
{
    PT pt;
    size_t offset = 0;

    // 辅助函数：读取整数
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

    // 读取content大小
    int content_size = read_int();
    pt.content.reserve(content_size);

    // 读取content数据
    for (int i = 0; i < content_size; ++i)
    {
        int type = read_int();
        int length = read_int();
        pt.content.emplace_back(type, length);
    }

    // 读取pivot
    pt.pivot = read_int();

    // 读取curr_indices
    int curr_size = read_int();
    pt.curr_indices.reserve(curr_size);
    for (int i = 0; i < curr_size; ++i)
    {
        pt.curr_indices.emplace_back(read_int());
    }

    // 读取max_indices
    int max_size = read_int();
    pt.max_indices.reserve(max_size);
    for (int i = 0; i < max_size; ++i)
    {
        pt.max_indices.emplace_back(read_int());
    }

    // 初始化浮点数为0（worker不需要）
    // pt.preterm_prob = 0.0f;
    // pt.prob = 0.0f;

    return pt;
}