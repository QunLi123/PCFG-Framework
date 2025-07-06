#include "PCFG.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
using namespace std;

// ✅ 统一预加载的GPU kernel（这是我们唯一需要的kernel）
__global__ void generateBatchUnifiedKernel(
    char *d_base_guesses_buffer, int *d_base_guess_lengths, int *d_base_guess_offsets,
    int *d_suffix_base_indices, // 每个PT的suffix在统一数组中的起始索引
    int *d_suffix_counts,       // 每个PT有多少个suffix
    char *d_unified_strings,    // 统一的字符串缓冲区
    int *d_unified_offsets,     // 统一的偏移数组
    int *d_unified_lengths,     // 统一的长度数组
    char *d_guesses_buffer, int *d_guess_offsets, int max_guess_len,
    int batch_size, int *d_pt_guess_counts)
{
    int pt_idx = blockIdx.y;
    int val_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pt_idx >= batch_size || val_idx >= d_suffix_counts[pt_idx])
        return;

    int guess_idx = d_guess_offsets[pt_idx] + val_idx;
    char *temp_guess = d_guesses_buffer + guess_idx * max_guess_len;

    // ✅ 拷贝base guess
    int base_len = d_base_guess_lengths[pt_idx];
    char *base_guess = d_base_guesses_buffer + d_base_guess_offsets[pt_idx];
    // for (int i = 0; i < base_len; i++) {
    //     temp_guess[i] = base_guess[i];
    // }
    if (base_len > 0)
    {
        memcpy(temp_guess, base_guess, base_len);
    }
    // ✅ 简化的suffix获取：一次计算，直接访问
    int unified_string_idx = d_suffix_base_indices[pt_idx] + val_idx;
    int string_offset = d_unified_offsets[unified_string_idx];
    int string_length = d_unified_lengths[unified_string_idx];

    // ✅ 拷贝suffix
    // for (int i = 0; i < string_length; i++)
    // {
    //     temp_guess[base_len + i] = d_unified_strings[string_offset + i];
    // }
    memcpy(temp_guess + base_len, d_unified_strings + string_offset, string_length);

    temp_guess[base_len + string_length] = '\0';

    atomicAdd(&d_pt_guess_counts[pt_idx], 1);
}

// ✅ CPU内存池初始化
void PriorityQueue::initCPUMemoryPool()
{
    if (cpu_pools_initialized)
        return;

    cout << "Initializing CPU memory pool..." << endl;

    h_output_pool.resize(1500000 * 32);
    h_pt_counts_pool.resize(32);
    h_values_buffer_pool.resize(1500000 * 32);
    h_value_lengths_pool.resize(1500000);
    h_value_offsets_pool.resize(33);
    h_base_guess_lengths_pool.resize(32);
    h_base_guess_offsets_pool.resize(33);
    h_guess_offsets_pool.resize(33);

    cpu_pools_initialized = true;
    cout << "CPU memory pool initialized!" << endl;
}

void PriorityQueue::cleanupCPUMemoryPool()
{
    if (!cpu_pools_initialized)
        return;

    cout << "Cleaning up CPU memory pool..." << endl;
    h_output_pool.clear();
    h_output_pool.shrink_to_fit();
    h_pt_counts_pool.clear();
    h_pt_counts_pool.shrink_to_fit();
    h_values_buffer_pool.clear();
    h_values_buffer_pool.shrink_to_fit();
    h_value_lengths_pool.clear();
    h_value_lengths_pool.shrink_to_fit();
    h_value_offsets_pool.clear();
    h_value_offsets_pool.shrink_to_fit();
    h_base_guess_lengths_pool.clear();
    h_base_guess_lengths_pool.shrink_to_fit();
    h_base_guess_offsets_pool.clear();
    h_base_guess_offsets_pool.shrink_to_fit();
    h_guess_offsets_pool.clear();
    h_guess_offsets_pool.shrink_to_fit();

    cpu_pools_initialized = false;
    cout << "CPU memory pool cleaned up." << endl;
}

// ✅ CPU局部变量池初始化
void PriorityQueue::initCPULocalVars()
{
    if (cpu_local_vars_initialized)
        return;

    cout << "Initializing CPU local variables pool..." << endl;

    base_guesses_pool.resize(MAX_BATCH_SIZE);
    max_indices_pool.resize(MAX_BATCH_SIZE);
    segments_pool.resize(MAX_BATCH_SIZE);

    for (int i = 0; i < MAX_BATCH_SIZE; ++i)
    {
        base_guesses_pool[i].reserve(64);
    }

    cpu_local_vars_initialized = true;
    cout << "CPU local variables pool initialized!" << endl;
}

void PriorityQueue::cleanupCPULocalVars()
{
    if (!cpu_local_vars_initialized)
        return;

    cout << "Cleaning up CPU local variables pool..." << endl;
    base_guesses_pool.clear();
    base_guesses_pool.shrink_to_fit();
    max_indices_pool.clear();
    max_indices_pool.shrink_to_fit();
    segments_pool.clear();
    segments_pool.shrink_to_fit();

    cpu_local_vars_initialized = false;
    cout << "CPU local variables pool cleaned up." << endl;
}

// ✅ GPU固定数组池初始化
void PriorityQueue::initGPUFixedArrays()
{
    if (gpu_fixed_arrays_initialized)
        return;

    cout << "Initializing GPU fixed arrays pool..." << endl;

    cudaMalloc(&d_value_lengths_pool, MAX_VALUES_PER_BATCH * sizeof(size_t));
    cudaMalloc(&d_value_offsets_pool, (MAX_BATCH_SIZE + 1) * sizeof(int));
    cudaMalloc(&d_max_indices_pool, MAX_BATCH_SIZE * sizeof(int));
    cudaMalloc(&d_base_lengths_pool, MAX_BATCH_SIZE * sizeof(int));
    cudaMalloc(&d_base_offsets_pool, (MAX_BATCH_SIZE + 1) * sizeof(int));
    cudaMalloc(&d_guess_offsets_pool, (MAX_BATCH_SIZE + 1) * sizeof(int));
    cudaMalloc(&d_pt_counts_pool, MAX_BATCH_SIZE * sizeof(int));

    gpu_fixed_arrays_initialized = true;
    cout << "GPU fixed arrays pool initialized!" << endl;
}

void PriorityQueue::cleanupGPUFixedArrays()
{
    if (!gpu_fixed_arrays_initialized)
        return;

    cout << "Cleaning up GPU fixed arrays pool..." << endl;
    if (d_value_lengths_pool)
    {
        cudaFree(d_value_lengths_pool);
        d_value_lengths_pool = nullptr;
    }
    if (d_value_offsets_pool)
    {
        cudaFree(d_value_offsets_pool);
        d_value_offsets_pool = nullptr;
    }
    if (d_max_indices_pool)
    {
        cudaFree(d_max_indices_pool);
        d_max_indices_pool = nullptr;
    }
    if (d_base_lengths_pool)
    {
        cudaFree(d_base_lengths_pool);
        d_base_lengths_pool = nullptr;
    }
    if (d_base_offsets_pool)
    {
        cudaFree(d_base_offsets_pool);
        d_base_offsets_pool = nullptr;
    }
    if (d_guess_offsets_pool)
    {
        cudaFree(d_guess_offsets_pool);
        d_guess_offsets_pool = nullptr;
    }
    if (d_pt_counts_pool)
    {
        cudaFree(d_pt_counts_pool);
        d_pt_counts_pool = nullptr;
    }

    gpu_fixed_arrays_initialized = false;
    cout << "GPU fixed arrays pool cleaned up." << endl;
}

// ✅ GPU字符缓冲区池初始化
void PriorityQueue::initGPUCharBuffers()
{
    if (gpu_char_buffers_initialized)
        return;

    cout << "Initializing GPU character buffers pool..." << endl;

    cudaMalloc(&d_values_buffer_pool, VALUES_BUFFER_SIZE);
    cudaMalloc(&d_base_buffer_pool, BASE_BUFFER_SIZE);
    cudaMalloc(&d_output_buffer_pool, OUTPUT_BUFFER_SIZE);

    gpu_char_buffers_initialized = true;
    cout << "GPU character buffers pool initialized!" << endl;
}

void PriorityQueue::cleanupGPUCharBuffers()
{
    if (!gpu_char_buffers_initialized)
        return;

    cout << "Cleaning up GPU character buffers pool..." << endl;
    if (d_values_buffer_pool)
    {
        cudaFree(d_values_buffer_pool);
        d_values_buffer_pool = nullptr;
    }
    if (d_base_buffer_pool)
    {
        cudaFree(d_base_buffer_pool);
        d_base_buffer_pool = nullptr;
    }
    if (d_output_buffer_pool)
    {
        cudaFree(d_output_buffer_pool);
        d_output_buffer_pool = nullptr;
    }

    gpu_char_buffers_initialized = false;
    cout << "GPU character buffers pool cleaned up." << endl;
}

// ✅ 统一预加载segment数据到GPU（这是核心函数）
void PriorityQueue::preloadUnifiedSegmentsToGPU()
{
    if (unified_segments_preloaded)
        return;

    cout << "Preloading unified segment data to GPU..." << endl;

    // ✅ 构建统一的字符串缓冲区和索引
    string all_unified_strings;
    vector<int> unified_offsets, unified_lengths;
    int current_offset = 0;
    int current_string_idx = 0;

    // 清空并重置辅助索引
    letters_base_indices.clear();
    letters_base_indices.resize(m.letters.size());
    digits_base_indices.clear();
    digits_base_indices.resize(m.digits.size());
    symbols_base_indices.clear();
    symbols_base_indices.resize(m.symbols.size());

    // ✅ 处理所有letters segments
    for (int seg_idx = 0; seg_idx < m.letters.size(); seg_idx++)
    {
        letters_base_indices[seg_idx].push_back(current_string_idx);

        for (const auto &value : m.letters[seg_idx].ordered_values)
        {
            unified_offsets.push_back(current_offset);
            unified_lengths.push_back(value.size());
            all_unified_strings += value + '\0';
            current_offset += value.size() + 1;
            current_string_idx++;
        }
    }

    // ✅ 处理所有digits segments
    for (int seg_idx = 0; seg_idx < m.digits.size(); seg_idx++)
    {
        digits_base_indices[seg_idx].push_back(current_string_idx);

        for (const auto &value : m.digits[seg_idx].ordered_values)
        {
            unified_offsets.push_back(current_offset);
            unified_lengths.push_back(value.size());
            all_unified_strings += value + '\0';
            current_offset += value.size() + 1;
            current_string_idx++;
        }
    }

    // ✅ 处理所有symbols segments
    for (int seg_idx = 0; seg_idx < m.symbols.size(); seg_idx++)
    {
        symbols_base_indices[seg_idx].push_back(current_string_idx);

        for (const auto &value : m.symbols[seg_idx].ordered_values)
        {
            unified_offsets.push_back(current_offset);
            unified_lengths.push_back(value.size());
            all_unified_strings += value + '\0';
            current_offset += value.size() + 1;
            current_string_idx++;
        }
    }

    // ✅ 分配GPU内存并传输统一数据
    cudaMalloc(&d_unified_strings, all_unified_strings.size());
    cudaMalloc(&d_unified_offsets, unified_offsets.size() * sizeof(int));
    cudaMalloc(&d_unified_lengths, unified_lengths.size() * sizeof(int));

    cudaMemcpy(d_unified_strings, all_unified_strings.c_str(), all_unified_strings.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_unified_offsets, unified_offsets.data(), unified_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_unified_lengths, unified_lengths.data(), unified_lengths.size() * sizeof(int), cudaMemcpyHostToDevice);

    unified_segments_preloaded = true;

    // ✅ 输出统计信息
    cout << "Unified segment data preloaded to GPU: " << all_unified_strings.size() / 1024.0 / 1024.0 << " MB" << endl;
    cout << "Total strings: " << unified_offsets.size() << endl;
    cout << "Letters segments: " << m.letters.size() << endl;
    cout << "Digits segments: " << m.digits.size() << endl;
    cout << "Symbols segments: " << m.symbols.size() << endl;
}

// ✅ 清理统一预加载的数据
void PriorityQueue::cleanupUnifiedSegments()
{
    if (!unified_segments_preloaded)
        return;

    cout << "Cleaning up unified segment data..." << endl;

    if (d_unified_strings)
    {
        cudaFree(d_unified_strings);
        d_unified_strings = nullptr;
    }
    if (d_unified_offsets)
    {
        cudaFree(d_unified_offsets);
        d_unified_offsets = nullptr;
    }
    if (d_unified_lengths)
    {
        cudaFree(d_unified_lengths);
        d_unified_lengths = nullptr;
    }

    unified_segments_preloaded = false;
    cout << "Unified segment data cleaned up." << endl;
}

// ✅ 概率计算
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
        else if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        else if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index++;
    }
}

// ✅ 初始化函数
void PriorityQueue::init()
{
    // 初始化所有内存池
    if (!cpu_pools_initialized)
    {
        initCPUMemoryPool();
    }
    if (!cpu_local_vars_initialized)
    {
        initCPULocalVars();
    }
    if (!gpu_fixed_arrays_initialized)
    {
        initGPUFixedArrays();
    }
    if (!gpu_char_buffers_initialized)
    {
        initGPUCharBuffers();
    }

    // 初始化PT数据
    for (PT &pt : m.ordered_pts)
    {
        for (const segment &seg : pt.content)
        {
            if (seg.type == 1)
            {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            else if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            else if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }

    // ✅ 预加载统一的segment数据
    preloadUnifiedSegmentsToGPU();
}

// ✅ PopNext函数
void PriorityQueue::PopNext()
{
    const int batch_size = 32;

    vector<PT> batch_pts(batch_size);
    for (int i = 0; i < batch_size; i++)
    {
        batch_pts[i] = priority.front();
        priority.erase(priority.begin());
    }

    Generate(batch_pts);

    for (PT &pt : batch_pts)
    {
        vector<PT> new_pts = pt.NewPTs();
        for (PT &new_pt : new_pts)
        {
            CalProb(new_pt);
            auto iter = priority.begin();
            while (iter != priority.end() && new_pt.prob <= iter->prob)
            {
                ++iter;
            }
            priority.insert(iter, new_pt);
        }
    }
    cudaDeviceSynchronize();
}

// ✅ 使用统一预加载的Generate函数（这是核心函数）
void PriorityQueue::Generate(vector<PT> &pts)
{
    int batch_size = 32;
    int total_values = 0, max_guess_len = 0;

    // ✅ 准备批处理数据：计算每个PT的suffix信息
    vector<int> suffix_base_indices(batch_size); // 每个PT的suffix在统一数组中的起始索引
    vector<int> suffix_counts(batch_size);       // 每个PT有多少个suffix

    for (int i = 0; i < batch_size; ++i)
    {
        PT &pt = pts[i];

        // 构建base guess
        base_guesses_pool[i].clear();
        for (int j = 0; j < pt.curr_indices.size(); ++j)
        {
            if (j == pt.content.size() - 1)
                break;
            int idx = pt.curr_indices[j];
            const segment &seg = pt.content[j];
            if (seg.type == 1)
                base_guesses_pool[i] += m.letters[m.FindLetter(seg)].ordered_values[idx];
            else if (seg.type == 2)
                base_guesses_pool[i] += m.digits[m.FindDigit(seg)].ordered_values[idx];
            else
                base_guesses_pool[i] += m.symbols[m.FindSymbol(seg)].ordered_values[idx];
        }

        // ✅ 计算最后一个segment的统一索引
        const segment &last_seg = pt.content.back();

        if (last_seg.type == 1)
        {
            int seg_idx = m.FindLetter(last_seg);
            suffix_base_indices[i] = letters_base_indices[seg_idx][0];
            suffix_counts[i] = m.letters[seg_idx].ordered_values.size();
        }
        else if (last_seg.type == 2)
        {
            int seg_idx = m.FindDigit(last_seg);
            suffix_base_indices[i] = digits_base_indices[seg_idx][0];
            suffix_counts[i] = m.digits[seg_idx].ordered_values.size();
        }
        else
        {
            int seg_idx = m.FindSymbol(last_seg);
            suffix_base_indices[i] = symbols_base_indices[seg_idx][0];
            suffix_counts[i] = m.symbols[seg_idx].ordered_values.size();
        }

        total_values += suffix_counts[i];
        max_indices_pool[i] = suffix_counts[i];

        // 估算最大guess长度
        if (last_seg.type == 1)
        {
            int seg_idx = m.FindLetter(last_seg);
            for (const auto &val : m.letters[seg_idx].ordered_values)
            {
                max_guess_len = max(max_guess_len, (int)(base_guesses_pool[i].size() + val.size() + 1));
            }
        }
        else if (last_seg.type == 2)
        {
            int seg_idx = m.FindDigit(last_seg);
            for (const auto &val : m.digits[seg_idx].ordered_values)
            {
                max_guess_len = max(max_guess_len, (int)(base_guesses_pool[i].size() + val.size() + 1));
            }
        }
        else
        {
            int seg_idx = m.FindSymbol(last_seg);
            for (const auto &val : m.symbols[seg_idx].ordered_values)
            {
                max_guess_len = max(max_guess_len, (int)(base_guesses_pool[i].size() + val.size() + 1));
            }
        }
    }

    // ✅ 准备offset数组
    h_base_guess_offsets_pool[0] = 0;
    h_guess_offsets_pool[0] = 0;
    for (int i = 0; i < batch_size; ++i)
    {
        h_base_guess_lengths_pool[i] = base_guesses_pool[i].size();
        h_base_guess_offsets_pool[i + 1] = h_base_guess_offsets_pool[i] + base_guesses_pool[i].size() + 1;
        h_guess_offsets_pool[i + 1] = h_guess_offsets_pool[i] + suffix_counts[i];
    }

    // ✅ 分配临时GPU内存（suffix信息）
    int *d_suffix_base_indices, *d_suffix_counts;
    cudaMalloc(&d_suffix_base_indices, batch_size * sizeof(int));
    cudaMalloc(&d_suffix_counts, batch_size * sizeof(int));

    // ✅ 传输数据
    cudaMemset(d_pt_counts_pool, 0, batch_size * sizeof(int));
    cudaMemcpyAsync(d_suffix_base_indices, suffix_base_indices.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_suffix_counts, suffix_counts.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_base_lengths_pool, h_base_guess_lengths_pool.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_base_offsets_pool, h_base_guess_offsets_pool.data(), (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_guess_offsets_pool, h_guess_offsets_pool.data(), (batch_size + 1) * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < batch_size; ++i)
    {
        cudaMemcpy(d_base_buffer_pool + h_base_guess_offsets_pool[i], base_guesses_pool[i].c_str(),
                   base_guesses_pool[i].size() + 1, cudaMemcpyHostToDevice);
    }

    // ✅ 启动统一kernel
    dim3 block(256);
    int max_values_per_pt = *max_element(suffix_counts.begin(), suffix_counts.end());
    dim3 grid((max_values_per_pt + block.x - 1) / block.x, batch_size);

    generateBatchUnifiedKernel<<<grid, block>>>(
        d_base_buffer_pool, d_base_lengths_pool, d_base_offsets_pool,
        d_suffix_base_indices, d_suffix_counts,
        d_unified_strings, d_unified_offsets, d_unified_lengths,
        d_output_buffer_pool, d_guess_offsets_pool, max_guess_len,
        batch_size, d_pt_counts_pool);

    // ✅ 拷贝结果
    size_t output_needed = total_values * max_guess_len;
    cudaMemcpy(h_output_pool.data(), d_output_buffer_pool, output_needed, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pt_counts_pool.data(), d_pt_counts_pool, batch_size * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < total_values; ++i)
    {
        guesses.emplace_back(h_output_pool.data() + i * max_guess_len);
    }

    total_guesses += accumulate(h_pt_counts_pool.begin(), h_pt_counts_pool.begin() + batch_size, 0);

    // ✅ 释放临时GPU内存
    cudaFree(d_suffix_base_indices);
    cudaFree(d_suffix_counts);
}

// ✅ NewPTs函数保持不变
vector<PT> PT::NewPTs()
{
    vector<PT> res;
    // if (content.size() <= 1)
    //     return res;

    int init_pivot = pivot;
    for (int i = pivot; i < curr_indices.size() - 1; i++)
    {
        curr_indices[i]++;
        if (curr_indices[i] < max_indices[i])
        {
            pivot = i;
            res.emplace_back(*this);
        }
        curr_indices[i]--;
    }
    pivot = init_pivot;
    return res;
}