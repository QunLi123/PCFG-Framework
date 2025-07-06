#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
using namespace std;
using namespace chrono;

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2

int main()
{
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "here" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;
    // std::ofstream a("./files/results.txt");
    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();
        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated1 " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            // 在此处更改实验生成的猜测上限
            int generate_n = 10000000;
            if (history + q.total_guesses > 10000000)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                cout << "Hash time:" << time_hash << "seconds" << endl;
                cout << "Train time:" << time_train << "seconds" << endl;
                break;
            }
        }
        // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
        // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
        if (curr_num > 1000000)
        {
            auto start_hash = system_clock::now();
            //   bit32 state[4];
            //   for (string pw : q.guesses)
            //   {
            //       ////TODO：对于SIMD实验，将这里替换成你的SIMD MD5函数
            //       MD5Hash(pw, state);
            //   }

            // 以下注释部分用于输出猜测和哈希，但是由于自动测试系统不太能写文件，所以这里你可以改成cout
            // a<<pw<<"\t";
            // for (int i1 = 0; i1 < 4; i1 += 1)
            // {
            //     a << std::setw(8) << std::setfill('0') << hex << state[i1];
            // }
            // a << endl;
            //}
            // 预分配states数组
            //              const int batch_size = ((curr_num + 3) / 4) * 4; // 向上对齐到4的倍数
            //              bit32 (*states)[4] = new bit32 [batch_size][4];
            //              // 准备长度数组
            //              vector<int> lengths(curr_num);
            //  #pragma omp parallel for
            //              for (int i = 0; i < curr_num; i++)
            //              {
            //                  lengths[i] = q.guesses[i].length();
            //              }
            //              // 调用SIMD版本的MD5
            //              MD5Hash_SIMD(q.guesses, lengths.data(), curr_num, states);

               alignas(16) bit32 states[12][4];
               string strs[12];
               for (int i = 0; i < q.guesses.size(); i += 12)
               {
                   strs[0] = q.guesses[i];
                   strs[1] = i + 1 < q.guesses.size() ? q.guesses[i + 1] : q.guesses[i];
                   strs[2] = i + 2 < q.guesses.size() ? q.guesses[i + 2] : q.guesses[i];
                   strs[3] = i + 3 < q.guesses.size() ? q.guesses[i + 3] : q.guesses[i];
                   strs[4] = i + 4 < q.guesses.size() ? q.guesses[i + 4] : q.guesses[i];
                   strs[5] = i + 5 < q.guesses.size() ? q.guesses[i + 5] : q.guesses[i];
                   strs[6] = i + 6 < q.guesses.size() ? q.guesses[i + 6] : q.guesses[i];
                   strs[7] = i + 7 < q.guesses.size() ? q.guesses[i + 7] : q.guesses[i];
                   strs[8] = i + 8 < q.guesses.size() ? q.guesses[i + 8] : q.guesses[i];
                   strs[9] = i + 9 < q.guesses.size() ? q.guesses[i + 9] : q.guesses[i];
                   strs[10] = i + 10 < q.guesses.size() ? q.guesses[i + 10] : q.guesses[i];
                   strs[11] = i + 11 < q.guesses.size() ? q.guesses[i + 11] : q.guesses[i];
                //    strs[12] = i + 12 < q.guesses.size() ? q.guesses[i + 12] : q.guesses[i];
                //    strs[13] = i + 13 < q.guesses.size() ? q.guesses[i + 13] : q.guesses[i];
                //    strs[14] = i + 14 < q.guesses.size() ? q.guesses[i + 14] : q.guesses[i];
                //    strs[15] = i + 15 < q.guesses.size() ? q.guesses[i + 15] : q.guesses[i];
                   MD5Hash_SIMD12(strs, states);
                // bit32 state[4];
                // for (int j = 0; j < 4; j++)
                // {
                //     MD5Hash(strs[j], state);
                //     if (states[j][0] != state[0] || states[j][1] != state[1] || states[j][2] != state[2] || states[j][3] != state[3])
                //     {
                //         cout << "error" << ' ' << i << endl;
                //     }
                // }
              }
            // 在这里对哈希所需的总时长进行计算
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            // 记录已经生成的口令总数
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
}
