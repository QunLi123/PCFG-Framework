#include "PCFG.h"
#include <cstring>
#include <chrono>

using namespace std;

#ifdef USE_CUDA
// 声明CUDA函数
void cuda_generate_guesses(const char *flat_values, int value_len, 
                          const char *h_guess_prefix, int prefix_len, int n, 
                          std::vector<std::string> &guesses, size_t offset,
                          double* prepare_time, double* kernel_time, double* collect_time);
#endif

// 全局性能计时变量
double last_prepare_time = 0, last_kernel_time = 0, last_collect_time = 0;

void PriorityQueue::CalProb(PT &pt)
{
    // 计算PT的概率
    pt.prob = pt.preterm_prob;
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
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
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext()
{
    // 生成猜测
    Generate(priority.front());

    // 生成新的PT
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
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

    priority.erase(priority.begin());
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;

    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        int init_pivot = pivot;

        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            curr_indices[i] += 1;

            if (curr_indices[i] < max_indices[i])
            {
                pivot = i;
                res.emplace_back(*this);
            }

            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}

// 主要的密码生成函数 - PCFG并行化算法的核心
void PriorityQueue::Generate(PT pt)
{
    using namespace std::chrono;
    CalProb(pt);
    double t_prepare = 0, t_kernel = 0, t_collect = 0;
    
    if (pt.content.size() == 1)
    {
        // 单个segment的情况
        segment *current_seg;
        if (pt.content[0].type == 1)
            current_seg = &m.letters[m.FindLetter(pt.content[0])];
        if (pt.content[0].type == 2)
            current_seg = &m.digits[m.FindDigit(pt.content[0])];
        if (pt.content[0].type == 3)
            current_seg = &m.symbols[m.FindSymbol(pt.content[0])];
            
        int n = pt.max_indices[0];
        int value_len = 0;
        if (n > 0) value_len = current_seg->ordered_values[0].size();
        
        #ifdef USE_CUDA
        // 准备扁平化数据
        char *flat_values = new char[n * (value_len + 1)];
        for (int i = 0; i < n; ++i) {
            memcpy(flat_values + i * (value_len + 1), current_seg->ordered_values[i].c_str(), value_len + 1);
        }
        
        size_t old_size = guesses.size();
        guesses.resize(old_size + n);
        
        cuda_generate_guesses(flat_values, value_len, nullptr, 0, n, guesses, old_size, 
                             &t_prepare, &t_kernel, &t_collect);
        
        delete[] flat_values;
        total_guesses += n;
        
        last_prepare_time = t_prepare;
        last_kernel_time = t_kernel;
        last_collect_time = t_collect;
        #else
        // 串行版本
        std::vector<std::string> temp_guesses;
        for (int i = 0; i < n; i += 1)
        {
            temp_guesses.emplace_back(current_seg->ordered_values[i]);
        }
        for (const auto& s : temp_guesses) {
            guesses.emplace_back(s);
            total_guesses += 1;
        }
        #endif
    }
    else
    {
        // 多个segment的情况
        string guess_prefix;
        int seg_idx = 0;
        
        // 构建前缀
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
                guess_prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 2)
                guess_prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 3)
                guess_prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
                break;
        }
        
        // 处理最后一个segment
        segment *last_seg;
        if (pt.content[pt.content.size() - 1].type == 1)
            last_seg = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        if (pt.content[pt.content.size() - 1].type == 2)
            last_seg = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        if (pt.content[pt.content.size() - 1].type == 3)
            last_seg = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
            
        int n = pt.max_indices[pt.content.size() - 1];
        int value_len = 0;
        if (n > 0) value_len = last_seg->ordered_values[0].size();
        int prefix_len = guess_prefix.size();
        
        #ifdef USE_CUDA
        // 准备扁平化数据
        char *flat_values = new char[n * (value_len + 1)];
        for (int i = 0; i < n; ++i) {
            memcpy(flat_values + i * (value_len + 1), last_seg->ordered_values[i].c_str(), value_len + 1);
        }
        
        size_t old_size = guesses.size();
        guesses.resize(old_size + n);
        
        cuda_generate_guesses(flat_values, value_len, guess_prefix.c_str(), prefix_len, n, 
                             guesses, old_size, &t_prepare, &t_kernel, &t_collect);
        
        delete[] flat_values;
        total_guesses += n;
        
        last_prepare_time = t_prepare;
        last_kernel_time = t_kernel;
        last_collect_time = t_collect;
        #else
        // 串行版本
        std::vector<std::string> temp_guesses;
        for (int i = 0; i < n; i += 1)
        {
            string temp = guess_prefix + last_seg->ordered_values[i];
            temp_guesses.emplace_back(temp);
        }
        for (const auto& s : temp_guesses) {
            guesses.emplace_back(s);
            total_guesses += 1;
        }
        #endif
    }
}