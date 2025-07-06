#include <string>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <omp.h>
// #include <chrono>
// using namespace chrono;
using namespace std;
class segment
{
public:
    int type;   // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    int length; // 长度，例如S6的长度就是6
    segment(int type, int length)
    {
        this->type = type;
        this->length = length;
    };

    // 打印相关信息
    void PrintSeg();

    // 按照概率降序排列的value。例如，123是D3的一个具体value，其概率在D3的所有value中排名第三，那么其位置就是ordered_values[2]
    vector<string> ordered_values;

    // 按照概率降序排列的频数（概率）
    vector<int> ordered_freqs;

    // total_freq作为分母，用于计算每个value的概率
    int total_freq = 0;

    // 未排序的value，其中int就是对应的id
    unordered_map<string, int> values;

    // 根据id，在freqs中查找/修改一个value的频数
    unordered_map<int, int> freqs;

    void insert(string value);
    void order();
    void PrintValues();
};

class PT
{
public:
    // 例如，L6D1的content大小为2，content[0]为L6，content[1]为D1
    vector<segment> content;

    // pivot值，参见PCFG的原理
    int pivot = 0;
    void insert(segment seg);
    void PrintPT();

    // 导出新的PT
    vector<PT> NewPTs();

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的下标
    vector<int> curr_indices;

    // 记录当前每个segment（除了最后一个）对应的value，在模型中的最大下标（即最大可以是max_indices[x]-1）
    vector<int> max_indices;
    // void init();
    float preterm_prob;
    float prob;
};

class model
{
public:
    // 对于PT/LDS而言，序号是递增的
    // 训练时每遇到一个新的PT/LDS，就获取一个新的序号，并且当前序号递增1
    int preterm_id = -1;
    int letters_id = -1;
    int digits_id = -1;
    int symbols_id = -1;
    int GetNextPretermID()
    {
        preterm_id++;
        return preterm_id;
    };
    int GetNextLettersID()
    {
        letters_id++;
        return letters_id;
    };
    int GetNextDigitsID()
    {
        digits_id++;
        return digits_id;
    };
    int GetNextSymbolsID()
    {
        symbols_id++;
        return symbols_id;
    };

    // C++上机和数据结构实验中，一般不允许使用stl
    // 这就导致大家对stl不甚熟悉。现在是时候体会stl的便捷之处了
    // unordered_map: 无序映射
    int total_preterm = 0;
    vector<PT> preterminals;
    int FindPT(PT pt);

    vector<segment> letters;
    vector<segment> digits;
    vector<segment> symbols;
    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);

    unordered_map<int, int> preterm_freq;
    unordered_map<int, int> letters_freq;
    unordered_map<int, int> digits_freq;
    unordered_map<int, int> symbols_freq;

    vector<PT> ordered_pts;

    // 给定一个训练集，对模型进行训练
    void train(string train_path);

    // 对已经训练的模型进行保存
    void store(string store_path);

    // 从现有的模型文件中加载模型
    void load(string load_path);

    // 对一个给定的口令进行切分
    void parse(string pw);

    void order();

    // 打印模型
    void print();
};

// 优先队列，用于按照概率降序生成口令猜测
// 实际上，这个class负责队列维护、口令生成、结果存储的全部过程
class PriorityQueue
{
public:
    vector<PT> priority;
    model m;
    void CalProb(PT &pt);
    void init();
    void Generate(vector<PT> &pts);
    void PopNext();
    int total_guesses = 0;
    vector<string> guesses;

    // ✅ CPU端内存池
    vector<char> h_output_pool;
    vector<int> h_pt_counts_pool;
    vector<char> h_values_buffer_pool;
    vector<size_t> h_value_lengths_pool;
    vector<int> h_value_offsets_pool;
    vector<int> h_base_guess_lengths_pool;
    vector<int> h_base_guess_offsets_pool;
    vector<int> h_guess_offsets_pool;
    bool cpu_pools_initialized = false;

    // ✅ CPU局部变量池
    vector<string> base_guesses_pool;
    vector<int> max_indices_pool;
    vector<segment *> segments_pool;
    bool cpu_local_vars_initialized = false;

    // ✅ GPU固定数组池
    size_t *d_value_lengths_pool = nullptr;
    int *d_value_offsets_pool = nullptr;
    int *d_max_indices_pool = nullptr;
    int *d_base_lengths_pool = nullptr;
    int *d_base_offsets_pool = nullptr;
    int *d_guess_offsets_pool = nullptr;
    int *d_pt_counts_pool = nullptr;
    bool gpu_fixed_arrays_initialized = false;

    // ✅ GPU字符缓冲区池
    char *d_values_buffer_pool = nullptr;
    char *d_base_buffer_pool = nullptr;
    char *d_output_buffer_pool = nullptr;
    bool gpu_char_buffers_initialized = false;

    // ✅ 统一的GPU预加载数据（这是我们现在使用的）
    char *d_unified_strings = nullptr; // 所有segment字符串的统一缓冲区
    int *d_unified_offsets = nullptr;  // 每个字符串在缓冲区中的偏移
    int *d_unified_lengths = nullptr;  // 每个字符串的长度

    // ✅ CPU端辅助数据：用于快速计算偏移
    vector<vector<int>> letters_base_indices; // letters[i] 在统一数组中的起始索引
    vector<vector<int>> digits_base_indices;  // digits[i] 在统一数组中的起始索引
    vector<vector<int>> symbols_base_indices; // symbols[i] 在统一数组中的起始索引

    bool unified_segments_preloaded = false;

    // ✅ 内存池和预加载管理函数
    void initCPUMemoryPool();
    void cleanupCPUMemoryPool();
    void initCPULocalVars();
    void cleanupCPULocalVars();
    void initGPUFixedArrays();
    void cleanupGPUFixedArrays();
    void initGPUCharBuffers();
    void cleanupGPUCharBuffers();
    void preloadUnifiedSegmentsToGPU();
    void cleanupUnifiedSegments();

    // ✅ 常量定义
    static const int MAX_BATCH_SIZE = 32;
    static const int MAX_VALUES_PER_BATCH = 1500000;
    static const size_t VALUES_BUFFER_SIZE = 16 * 1024 * 1024;
    static const size_t BASE_BUFFER_SIZE = 2 * 1024 * 1024;
    static const size_t OUTPUT_BUFFER_SIZE = 16 * 1024 * 1024;
    // 添加缓存下一批数据的成员变量
    vector<PT> cached_next_batch;
    vector<string> cached_base_guesses;
    vector<int> cached_suffix_base_indices;
    vector<int> cached_suffix_counts;
    bool has_cached_batch = false;
    void precompute_next_batch_metadata(const vector<PT>& next_pts);



     // ✅ 添加用于processGPUResults的成员变量
    int current_total_values = 0;
    int current_max_guess_len = 0;

    // ✅ 添加函数声明
    void generateWithCachedMetadata(vector<PT> &pts);
    void processGPUResults();
    ~PriorityQueue()
    {
        cleanupUnifiedSegments();
        cleanupGPUCharBuffers();
        cleanupGPUFixedArrays();
        cleanupCPULocalVars();
        cleanupCPUMemoryPool();
    }
    PriorityQueue() = default;
};