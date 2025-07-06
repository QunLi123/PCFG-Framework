#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
#include <cmath>
using namespace std;
using namespace chrono;

// 这是仿照多线程思路版本代码

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe -O1
// mpic++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe -O2
// mpirun -np 2 ./test.exe
// mpirun -np 4 ./test.exe
// gdb ./test.exe
//(gdb) run
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "here" << endl;

    const int SYNC_BATCH_SIZE = 60;
    int sync_counter = 0;

    int generate_n = 10000000;
    int global_continue = 1; // 所有进程是否继续的标志
    int curr_num = 0;
    int history = 0;
    auto start = system_clock::now();
    while (global_continue)
    {

        q.PopNext();
        if (++sync_counter % SYNC_BATCH_SIZE == 0)
        {
            MPI_Barrier(MPI_COMM_WORLD);
        }
        // MPI_Barrier(MPI_COMM_WORLD);

        // 所有进程更新 total_guesses 记录
        q.total_guesses = q.guesses.size();

        if (q.total_guesses - curr_num >= 100000)
        {
            curr_num = q.total_guesses;
        }

        int limit = 0;
        if (rank == 0 && (history + q.total_guesses >= generate_n))
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            auto end = system_clock::now();
            auto duration = duration_cast<microseconds>(end - start);
            time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
            cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
            cout << "Hash time: " << time_hash << " seconds" << endl;
            cout << "Train time: " << time_train << " seconds" << endl;
            limit = 1;
        }

        // 广播终止信号给所有进程
        MPI_Bcast(&limit, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 如果需要退出所有进程，则退出循环
        if (limit)
        {
            break;
        }

        // 执行哈希阶段（防止内存爆炸）
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

            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }

        // 再次用 MPI_Allreduce 判断是否还有进程需要继续
        // int local_flag = !q.priority.empty();
        // MPI_Allreduce(&local_flag, &global_continue, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
