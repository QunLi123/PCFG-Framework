#include <string>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

class segment
{
public:
    int type;    // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    int length;  // segment长度
    
    segment(int type, int length)
    {
        this->type = type;
        this->length = length;
    };

    // 打印segment信息
    void PrintSeg();

    // 按概率降序排列的值
    vector<string> ordered_values;

    // 按概率降序排列的频数
    vector<int> ordered_freqs;

    // 总频数，用于概率计算
    int total_freq = 0;

    // 未排序的值及其ID映射
    unordered_map<string, int> values;

    // 根据ID查找频数
    unordered_map<int, int> freqs;

    void insert(string value);
    void order();
    void PrintValues();
};

class PT
{
public:
    // PT的segment组成
    vector<segment> content;

    // pivot值，用于新PT生成
    int pivot = 0;
    
    void insert(segment seg);
    void PrintPT();

    // 生成新的PT
    vector<PT> NewPTs();

    // 当前各segment的值索引（除最后一个）
    vector<int> curr_indices;

    // 各segment的最大索引值
    vector<int> max_indices;
    
    float preterm_prob;  // PT本身的概率
    float prob;          // 完整概率
};

class model
{
public:
    // ID生成器
    int preterm_id = -1;
    int letters_id = -1;
    int digits_id = -1;
    int symbols_id = -1;
    
    int GetNextPretermID() { preterm_id++; return preterm_id; };
    int GetNextLettersID() { letters_id++; return letters_id; };
    int GetNextDigitsID() { digits_id++; return digits_id; };
    int GetNextSymbolsID() { symbols_id++; return symbols_id; };

    // 模型数据存储
    int total_preterm = 0;
    vector<PT> preterminals;
    int FindPT(PT pt);

    vector<segment> letters;
    vector<segment> digits;
    vector<segment> symbols;
    
    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);

    // 频数映射
    unordered_map<int, int> preterm_freq;
    unordered_map<int, int> letters_freq;
    unordered_map<int, int> digits_freq;
    unordered_map<int, int> symbols_freq;

    vector<PT> ordered_pts;

    // 模型操作函数
    void train(string train_path);
    void store(string store_path);
    void load(string load_path);
    void parse(string pw);
    void order();
    void print();
};

// 优先队列类 - 负责密码生成的核心逻辑
class PriorityQueue
{
public:
    // 优先队列实现
    vector<PT> priority;

    // 模型实例
    model m;

    // 核心函数
    void CalProb(PT &pt);           // 计算PT概率
    void init();                    // 初始化队列
    void Generate(PT pt);           // 生成密码猜测
    void PopNext();                 // 处理队首PT

    // 状态变量
    int total_guesses = 0;          // 总生成数量
    vector<string> guesses;         // 密码存储
};