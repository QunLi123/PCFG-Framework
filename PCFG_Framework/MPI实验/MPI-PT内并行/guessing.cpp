#include "PCFG.h"
#include <mpi.h>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <cmath>
using namespace std;

void PriorityQueue::init_mpi_info()
{
    if (mpi_rank == -1)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    }
}

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    init_mpi_info();
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

void PriorityQueue::PopNext()
{

    // 对优先队列最前面的PT，首先利用这个PT生成一系列猜测
    Generate(priority.front());

    // 然后需要根据即将出队的PT，生成一系列新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        // 计算概率
        CalProb(pt);
        // 接下来的这个循环，作用是根据概率，将新的PT插入到优先队列中
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            // 对于非队首和队尾的特殊情况
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                // 判定概率
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}

// collect函数
void CollectGuesses(const vector<string> &local_guesses, int rank, int size, vector<string> &guesses,
                    int &total_guesses, int total_work)
{
    //int local_count = local_guesses.size();//各进程猜测数
    int local_count = total_work / size + (rank < total_work % size ? 1 : 0);//各进程猜测数 优化
    vector<int> all_counts(size);//进程猜测数统计
    // MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);优化
    int base = total_work / size;
    int remainder = total_work % size;
    for (int i = 0; i < size; ++i)
    {
        all_counts[i] = base + (i < remainder ? 1 : 0);
    }
    int first_length = local_guesses[0].size();
    vector<int> local_guess_lengths(local_count,first_length);//各进程猜测长度 优化
    
    // for (int i = 0; i < local_count; ++i)
    // {
    //     local_guess_lengths[i] = local_guesses[i].size();
    // }

    //vector<int> all_guess_lengths;
    vector<int> all_guess_lengths(total_work,first_length);//  优化
    //vector<int> length_displs(size);
    //int total_guesses_count = 0;优化
    //if (rank == 0)
    //{
        //total_guesses_count = accumulate(all_counts.begin(), all_counts.end(), 0);
        //all_guess_lengths.resize(total_work);
        
        // length_displs[0] = 0;
        // for (int i = 1; i < size; ++i)
        // {
        //     length_displs[i] = length_displs[i - 1] + all_counts[i - 1];
        // }
    //}
    // MPI_Gatherv(local_guess_lengths.data(), local_count, MPI_INT,
    //             all_guess_lengths.data(), all_counts.data(), length_displs.data(), MPI_INT,
    //             0, MPI_COMM_WORLD);
    string local_buffer;
    local_buffer.reserve(local_count * first_length);//优化
    for (const auto &s : local_guesses)
    {
        local_buffer += s;
    }
    //int local_buf_size = local_buffer.size();
    int local_buf_size = local_count * first_length; // 优化
    vector<int> recv_sizes(size);// 存储各进程的缓冲区大小
    //MPI_Gather(&local_buf_size, 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    vector<int> displs(size); // 存储各进程的偏移量
    // int total_size = 0;
    // if (rank == 0)
    // {
    //     displs[0] = 0;
    //     for (int i = 0; i < size; ++i)
    //     {
    //         displs[i] = total_size;
    //         total_size += recv_sizes[i];
    //     }
    // }
    // 优化
    if (rank == 0)
    {
        int total_size = 0;
        for (int i = 0; i < size; ++i)
        {
            recv_sizes[i] = all_counts[i] * first_length;  // 直接计算
            displs[i] = total_size;
            total_size += recv_sizes[i];
        }
    }
    vector<char> recv_buf;
    if (rank == 0)
    {
        recv_buf.resize(total_work * first_length);// 优化
    }
    MPI_Gatherv(local_buffer.data(), local_buf_size, MPI_CHAR,
                recv_buf.data(), recv_sizes.data(), displs.data(), MPI_CHAR,
                0, MPI_COMM_WORLD);
    // 根进程解析数据
    if (rank == 0)
    {
        string merged(recv_buf.begin(), recv_buf.end());
        // size_t offset = 0;
        // for (int len : all_guess_lengths)
        // {
        //     guesses.emplace_back(merged.substr(offset, len));
        //     offset += len;
        // }
        for (int i = 0; i < total_work; ++i)
        {
            size_t offset = i * first_length;
            guesses.emplace_back(merged.substr(offset, first_length));
        }
        total_guesses = guesses.size();
    }
}



void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    int rank = mpi_rank;
    int size = mpi_size;

    if (pt.content.size() == 1)
    {
        segment *a = nullptr;
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        else if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        else if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        // 分配任务
        // size进程数 rank进程号
        int total = pt.max_indices[0];
        int base = total / size;
        int remainder = total % size;
        int start = rank * base + min(rank, remainder);
        int end = start + base + (rank < remainder ? 1 : 0);

        vector<string> local_guesses;
        local_guesses.reserve(end - start);
        for (int i = start; i < end; ++i)
        {
            local_guesses.emplace_back(a->ordered_values[i]);
        }

        CollectGuesses(local_guesses, rank, size, guesses, total_guesses, total);
    }
    else
    {
        // prefix
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            else if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            else if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 定位最后一个 segment
        segment *a = nullptr;
        const int last_idx = pt.content.size() - 1;
        if (pt.content[last_idx].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[last_idx])];
        }
        else if (pt.content[last_idx].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[last_idx])];
        }
        else if (pt.content[last_idx].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[last_idx])];
        }

        int total = pt.max_indices[last_idx];
        int base = total / size;
        int remainder = total % size;
        int start = rank * base + min(rank, remainder);
        int end = start + base + (rank < remainder ? 1 : 0);

        // 生成本地猜测
        vector<string> local_guesses;
        local_guesses.reserve(end - start);
        for (int i = start; i < end; ++i)
        {

            local_guesses.emplace_back(guess + a->ordered_values[i]);
        }

        CollectGuesses(local_guesses, rank, size, guesses, total_guesses, total);
    }
}
