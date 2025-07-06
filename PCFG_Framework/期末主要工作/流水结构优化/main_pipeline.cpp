#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
#include <queue>
using namespace std;
using namespace chrono;

// 定义消息标签
#define TAG_PASSWORD_BATCH_SIZE 102
#define TAG_PASSWORD_DATA 103
#define TAG_TERMINATE 104
#define TAG_HASH_RESULT 105

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 3)
    {
        if (rank == 0)
        {
            cout << "This implementation requires exactly 3 processes!" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    double time_hash1 = 0, time_hash2 = 0, time_guess = 0, time_train = 0;
    const int PASSWORD_LIMIT = 10000000; // 1000万密码上限

    if (rank == 0)
    {
        // ========== 进程0: 密码生成器 ==========
        PriorityQueue q;

        auto start_train = system_clock::now();
        q.m.train("../Rockyou-singleLined-full.txt");
        q.m.order();
        auto end_train = system_clock::now();
        auto duration_train = duration_cast<microseconds>(end_train - start_train);
        time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

        q.init();

        int total_passwords_generated = 0;
        auto start_password_generation = system_clock::now();

        // 密码批次参数
        const int PASSWORD_BATCH_SIZE = 250000;
        vector<string> password_buffer;
        password_buffer.reserve(PASSWORD_BATCH_SIZE * 2);

        int next_hash_process = 1;

        while (total_passwords_generated < PASSWORD_LIMIT && !q.priority.empty())
        {
            // 密码生成
            vector<string> pt_passwords;
            while (password_buffer.size() < PASSWORD_BATCH_SIZE && !q.priority.empty() &&
                   total_passwords_generated < PASSWORD_LIMIT)
            {
                PT current_pt = q.priority.front();
                q.priority.erase(q.priority.begin());

                // 生成当前PT的密码
                pt_passwords.clear();
                q.Generate(current_pt, pt_passwords);

                // 添加到缓冲区
                for (const string &pw : pt_passwords)
                {
                    password_buffer.push_back(pw);
                    total_passwords_generated++;

                    if (total_passwords_generated >= PASSWORD_LIMIT)
                    {
                        break;
                    }
                }

                // 生成新的PT并插入队列
                vector<PT> new_pts = current_pt.NewPTs();
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

            // 当缓冲区足够时，发送给哈希进程
            if (password_buffer.size() >= PASSWORD_BATCH_SIZE || total_passwords_generated >= PASSWORD_LIMIT)
            {
                // 轮询发送给哈希进程
                int target_process = next_hash_process;
                next_hash_process = (next_hash_process == 1) ? 2 : 1;

                // 发送密码批次给目标哈希进程
                int batch_size = min((int)password_buffer.size(), PASSWORD_BATCH_SIZE);
                MPI_Send(&batch_size, 1, MPI_INT, target_process, TAG_PASSWORD_BATCH_SIZE, MPI_COMM_WORLD);

                // 序列化密码数据
                vector<char> pw_serialized_data;
                for (int i = 0; i < batch_size; i++)
                {
                    int pw_len = password_buffer[i].length();
                    pw_serialized_data.insert(pw_serialized_data.end(),
                                              reinterpret_cast<const char *>(&pw_len),
                                              reinterpret_cast<const char *>(&pw_len) + sizeof(int));
                    pw_serialized_data.insert(pw_serialized_data.end(),
                                              password_buffer[i].begin(),
                                              password_buffer[i].end());
                }

                int pw_data_size = pw_serialized_data.size();
                MPI_Send(&pw_data_size, 1, MPI_INT, target_process, TAG_PASSWORD_DATA, MPI_COMM_WORLD);
                MPI_Send(pw_serialized_data.data(), pw_data_size, MPI_CHAR, target_process, TAG_PASSWORD_DATA, MPI_COMM_WORLD);

                // 移除已发送的密码
                password_buffer.erase(password_buffer.begin(), password_buffer.begin() + batch_size);
            }
        }

        // 发送剩余的密码
        if (!password_buffer.empty())
        {
            int target_process = next_hash_process;
            int batch_size = password_buffer.size();
            MPI_Send(&batch_size, 1, MPI_INT, target_process, TAG_PASSWORD_BATCH_SIZE, MPI_COMM_WORLD);

            vector<char> pw_serialized_data;
            for (const string &password : password_buffer)
            {
                int pw_len = password.length();
                pw_serialized_data.insert(pw_serialized_data.end(),
                                          reinterpret_cast<const char *>(&pw_len),
                                          reinterpret_cast<const char *>(&pw_len) + sizeof(int));
                pw_serialized_data.insert(pw_serialized_data.end(), password.begin(), password.end());
            }

            int pw_data_size = pw_serialized_data.size();
            MPI_Send(&pw_data_size, 1, MPI_INT, target_process, TAG_PASSWORD_DATA, MPI_COMM_WORLD);
            MPI_Send(pw_serialized_data.data(), pw_data_size, MPI_CHAR, target_process, TAG_PASSWORD_DATA, MPI_COMM_WORLD);
        }

        // 发送终止信号给两个哈希进程
        int terminate_signal = -1;
        MPI_Send(&terminate_signal, 1, MPI_INT, 1, TAG_TERMINATE, MPI_COMM_WORLD);
        MPI_Send(&terminate_signal, 1, MPI_INT, 2, TAG_TERMINATE, MPI_COMM_WORLD);

        auto end_password_generation = system_clock::now();
        auto duration_guess = duration_cast<microseconds>(end_password_generation - start_password_generation);
        time_guess = double(duration_guess.count()) * microseconds::period::num / microseconds::period::den;

        // 接收两个哈希进程的时间统计
        MPI_Recv(&time_hash1, 1, MPI_DOUBLE, 1, TAG_HASH_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&time_hash2, 1, MPI_DOUBLE, 2, TAG_HASH_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // 计算平均哈希时间（模拟单线程总哈希时间）
        double total_hash_time = time_hash1 + time_hash2;

        // 与main.cpp保持一致的输出格式
        cout << "Guess time:" << time_guess - total_hash_time << "seconds" << endl;
        cout << "Hash time:" << total_hash_time << "seconds" << endl;
        cout << "Train time:" << time_train << "seconds" << endl;
    }
    else if (rank == 1 || rank == 2)
    {
        // ========== 进程1和2: 哈希计算器 ==========

        int my_hash_id = rank;
        int total_hashed = 0;
        double total_hash_time = 0.0; // 累积纯哈希时间

        while (true)
        {
            // 接收密码批次大小
            int password_batch_size;
            MPI_Status status;
            MPI_Recv(&password_batch_size, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_TERMINATE)
            {
                break;
            }

            if (status.MPI_TAG == TAG_PASSWORD_BATCH_SIZE && password_batch_size > 0)
            {
                // 接收密码数据
                int data_size;
                MPI_Recv(&data_size, 1, MPI_INT, 0, TAG_PASSWORD_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                vector<char> serialized_data(data_size);
                MPI_Recv(serialized_data.data(), data_size, MPI_CHAR, 0, TAG_PASSWORD_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // 反序列化密码
                vector<string> passwords;
                passwords.reserve(password_batch_size);

                size_t offset = 0;
                for (int i = 0; i < password_batch_size; ++i)
                {
                    int pw_len = *reinterpret_cast<const int *>(serialized_data.data() + offset);
                    offset += sizeof(int);

                    string password(serialized_data.data() + offset, pw_len);
                    offset += pw_len;

                    passwords.push_back(move(password));
                }

                // ========== 纯哈希计算时间 ==========
                auto start_hash_batch = system_clock::now();

                // 使用12路SIMD并行哈希
                alignas(16) bit32 states[12][4];
                string strs[12];

                for (int i = 0; i < passwords.size(); i += 12)
                {
                    // 准备12个字符串进行SIMD计算
                    for (int j = 0; j < 12; j++)
                    {
                        if (i + j < passwords.size())
                        {
                            strs[j] = passwords[i + j];
                        }
                        else
                        {
                            strs[j] = passwords[i]; // 用第一个密码填充
                        }
                    }

                    // 执行12路并行MD5哈希
                    MD5Hash_SIMD12(strs, states);
                }

                auto end_hash_batch = system_clock::now();
                // ========== 哈希时间计算结束 ==========

                auto duration_batch = duration_cast<microseconds>(end_hash_batch - start_hash_batch);
                double batch_time = double(duration_batch.count()) * microseconds::period::num / microseconds::period::den;
                total_hash_time += batch_time;

                total_hashed += password_batch_size;
            }
        }

        // 发送累积的纯哈希时间给进程0
        MPI_Send(&total_hash_time, 1, MPI_DOUBLE, 0, TAG_HASH_RESULT, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

// 序列化和反序列化函数保持不变
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