#include "PCFG.h"
#include <fstream>
using namespace std;

void PriorityQueue::CalProb(PT &pt)
{
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
    Generate(priority.front());
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

// 参考Generate_OpenMp_Simple优化的Generate函数
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        
        int n = pt.max_indices[0];
        size_t old_size = guesses.size();
        guesses.resize(old_size + n);
        
        if (n < 1150) {
            // 小任务串行执行
            for (int i = 0; i < n; i++) {
                guesses[old_size + i] = a->ordered_values[i];
            }
        } else {
            // 简化的OpenMP并行for循环
            #pragma omp parallel for num_threads(4) schedule(dynamic, 1000)
            for (int i = 0; i < n; i++) {
                guesses[old_size + i] = a->ordered_values[i];
            }
        }
        total_guesses += n;
    }
    else
    {
        string prefix;
        int seg_idx = 0;
        
        // 构建前缀
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
                prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            else
                prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
                break;
        }

        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        else if (pt.content[pt.content.size() - 1].type == 2)
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        
        int n = pt.max_indices[pt.content.size() - 1];
        int old_size = guesses.size();
        guesses.resize(old_size + n);
        
        if (n < 1150) {
            // 小任务串行执行
            for (int i = 0; i < n; i++) {
                guesses[old_size + i] = prefix + a->ordered_values[i];
            }
        } else {
            // 简化的OpenMP并行for循环
            #pragma omp parallel for num_threads(4) schedule(dynamic, 1000)
            for (int i = 0; i < n; i++) {
                guesses[old_size + i] = prefix;
                guesses[old_size + i] += a->ordered_values[i];
            }
        }
        total_guesses += n;
    }
}

// 参考Generate_OpenMp_Simple优化的Generate函数（带输出参数版本）
void PriorityQueue::Generate(const PT& pt, std::vector<std::string>& out_guesses)
{
    if (pt.content.size() == 1) {
        segment *a;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        
        int n = pt.max_indices[0];
        size_t old_size = out_guesses.size();
        out_guesses.resize(old_size + n);
        
        if (n < 1150) {
            // 小任务串行执行
            for (int i = 0; i < n; i++) {
                out_guesses[old_size + i] = a->ordered_values[i];
            }
        } else {
            // 简化的OpenMP并行for循环
            #pragma omp parallel for num_threads(4) schedule(dynamic, 1000)
            for (int i = 0; i < n; i++) {
                out_guesses[old_size + i] = a->ordered_values[i];
            }
        }
    } else {
        std::string prefix;
        int seg_idx = 0;
        
        // 构建前缀字符串（串行）
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1)
                prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            else
                prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
                break;
        }
        
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        else if (pt.content[pt.content.size() - 1].type == 2)
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        
        int n = pt.max_indices[pt.content.size() - 1];
        size_t old_size = out_guesses.size();
        out_guesses.resize(old_size + n);
        
        if (n < 1150) {
            // 小任务串行执行
            for (int i = 0; i < n; i++) {
                out_guesses[old_size + i] = prefix + a->ordered_values[i];
            }
        } else {
            // 简化的OpenMP并行for循环
            #pragma omp parallel for num_threads(4) schedule(dynamic, 1000)
            for (int i = 0; i < n; i++) {
                out_guesses[old_size + i] = prefix;
                out_guesses[old_size + i] += a->ordered_values[i];
            }
        }
    }
}