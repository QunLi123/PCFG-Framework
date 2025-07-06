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
#define TAG_BATCH_SIZE 100
#define TAG_PASSWORD_DATA 101
#define TAG_TERMINATE 102

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            cout << "This pipeline implementation requires exactly 2 processes!" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    double time_hash = 0, time_guess = 0, time_train = 0;
    
    if (rank == 0) {
        // ========== 进程0: 密码生成器 ==========
        PriorityQueue q;
        
        auto start_train = system_clock::now();
        q.m.train("./原始版本/Rockyou-singleLined-full.txt");
        q.m.order();
        auto end_train = system_clock::now();
        auto duration_train = duration_cast<microseconds>(end_train - start_train);
        time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

        q.init();

        int total_generated = 0;
        int generate_limit = 10000000;
        auto start = system_clock::now();

        // 流水线缓冲区参数
        const int BUFFER_SIZE = 100000;  // 每次发送的密码数量
        vector<string> password_buffer;
        password_buffer.reserve(BUFFER_SIZE);

        while (total_generated < generate_limit) {
            // 生成一批密码
            password_buffer.clear();
            
            while (password_buffer.size() < BUFFER_SIZE && total_generated < generate_limit) {
                PT current_pt = q.priority.front();
                q.priority.erase(q.priority.begin());
                
                // 生成当前PT的所有密码
                vector<string> pt_guesses;
                q.Generate(current_pt, pt_guesses);
                
                // 添加到缓冲区
                for (const string& guess : pt_guesses) {
                    if (password_buffer.size() >= BUFFER_SIZE || total_generated >= generate_limit) {
                        break;
                    }
                    password_buffer.push_back(guess);
                    total_generated++;
                }
                
                // 生成新的PT并插入队列
                vector<PT> new_pts = current_pt.NewPTs();
                for (const PT &new_pt : new_pts) {
                    PT pt_copy = new_pt;
                    q.CalProb(pt_copy);

                    // 按概率插入
                    bool inserted = false;
                    for (auto iter = q.priority.begin(); iter != q.priority.end(); ++iter) {
                        if (pt_copy.prob > iter->prob) {
                            q.priority.insert(iter, pt_copy);
                            inserted = true;
                            break;
                        }
                    }
                    if (!inserted) {
                        q.priority.push_back(pt_copy);
                    }
                }
            }

            if (!password_buffer.empty()) {
                // 发送密码批次给哈希进程
                int batch_size = password_buffer.size();
                MPI_Send(&batch_size, 1, MPI_INT, 1, TAG_BATCH_SIZE, MPI_COMM_WORLD);

                // 序列化密码数据
                vector<char> serialized_data;
                for (const string& pw : password_buffer) {
                    // 长度前缀 + 密码内容
                    int pw_len = pw.length();
                    serialized_data.insert(serialized_data.end(), 
                                         reinterpret_cast<const char*>(&pw_len), 
                                         reinterpret_cast<const char*>(&pw_len) + sizeof(int));
                    serialized_data.insert(serialized_data.end(), pw.begin(), pw.end());
                }

                int data_size = serialized_data.size();
                MPI_Send(&data_size, 1, MPI_INT, 1, TAG_PASSWORD_DATA, MPI_COMM_WORLD);
                MPI_Send(serialized_data.data(), data_size, MPI_CHAR, 1, TAG_PASSWORD_DATA, MPI_COMM_WORLD);
            }
        }

        int terminate_signal = -1;
        MPI_Send(&terminate_signal, 1, MPI_INT, 1, TAG_TERMINATE, MPI_COMM_WORLD);

        auto end = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;

        MPI_Recv(&time_hash, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        cout << "Total generated: " << total_generated << endl;
        cout << "Train time: " << time_train << " seconds" << endl;
        cout << "Guess time: " << time_guess << " seconds" << endl;
        cout << "Hash time: " << time_hash << " seconds" << endl;

    } else if (rank == 1) {
        // ========== 进程1: 哈希计算器 ==========
        
        int total_hashed = 0;

        while (true) {
            // 接收批次大小
            int batch_size;
            MPI_Status status;
            MPI_Recv(&batch_size, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_TERMINATE) {
                break;
            }

            if (status.MPI_TAG == TAG_BATCH_SIZE && batch_size > 0) {
                // 接收密码数据
                int data_size;
                MPI_Recv(&data_size, 1, MPI_INT, 0, TAG_PASSWORD_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                vector<char> serialized_data(data_size);
                MPI_Recv(serialized_data.data(), data_size, MPI_CHAR, 0, TAG_PASSWORD_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // 反序列化密码
                vector<string> passwords;
                passwords.reserve(batch_size);
                
                size_t offset = 0;
                for (int i = 0; i < batch_size; ++i) {
                    int pw_len = *reinterpret_cast<const int*>(serialized_data.data() + offset);
                    offset += sizeof(int);
                    
                    string password(serialized_data.data() + offset, pw_len);
                    offset += pw_len;
                    
                    passwords.push_back(move(password));
                }

                // 执行MD5哈希计算
                auto start_hash_batch = system_clock::now();
                bit32 state[4];
                
                for (const string& pw : passwords) {
                    MD5Hash(pw, state);
                }
                
                auto end_hash_batch = system_clock::now();
                auto duration_batch = duration_cast<microseconds>(end_hash_batch - start_hash_batch);
                time_hash += double(duration_batch.count()) * microseconds::period::num / microseconds::period::den;
                
                total_hashed += batch_size;
            }
        }

        MPI_Send(&time_hash, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

