#include "PCFG.h"
#include <pthread.h>
using namespace std;
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
    // Generate(priority.front());

    // 使用原始方法处理
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
        // // 处理边界情况：比队首元素概率大
        // if (pt.prob > priority.front().prob) {
        //     priority.emplace(priority.begin(), pt);
        //     continue;
        // }
        //
        // // 处理边界情况：比队尾元素概率小/等于
        // if (pt.prob <= priority.back().prob) {
        //     priority.push_back(pt);
        //     continue;
        // }
        //
        // // 二分查找 - 找到第一个概率小于pt.prob的元素的前一个位置
        // int left = 0;
        // int right = priority.size() - 1;
        // while (left <= right) {
        //     int mid = left + (right - left) / 2;
        //
        //     if (priority[mid].prob >= pt.prob &&
        //         (mid == priority.size() - 1 || priority[mid + 1].prob < pt.prob)) {
        //         // 找到正确位置：当前位置>=pt.prob，下一个位置<pt.prob
        //         priority.emplace(priority.begin() + mid + 1, pt);
        //         break;
        //     } else if (priority[mid].prob < pt.prob) {
        //         // 当前位置概率太小，向左移动
        //         right = mid - 1;
        //     } else {
        //         // 当前位置概率太大，或者下一个位置概率仍然>=pt.prob，向右移动
        //         left = mid + 1;
        //     }
        // }
    }

    // 现在队首的PT善后工作已经结束，将其出队（删除）
    priority.erase(priority.begin());
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
// 完全遍历pt空间的做法还是不可取的（空间太大卡死）
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

// 这个函数是PCFG并行化算法的主要载体
// 尽量看懂，然后进行并行实现
void PriorityQueue::Generate(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);

    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        for (int i = 0; i < pt.max_indices[0]; i += 1)
        {
            string guess = a->ordered_values[i];
            // cout << guess << endl;
            guesses.emplace_back(guess);
            total_guesses += 1;
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        // 这个for循环你看不懂也没太大问题，并行算法不涉及这里的加速
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }

        // Multi-thread TODO：
        // 这个for循环就是你需要进行并行化的主要部分了，特别是在多线程&GPU编程任务中
        // 可以看到，这个循环本质上就是把模型中一个segment的所有value，赋值到PT中，形成一系列新的猜测
        // 这个过程是可以高度并行化的
        for (int i = 0; i < pt.max_indices[pt.content.size() - 1]; i += 1)
        {
            string temp = guess + a->ordered_values[i];
            // cout << temp << endl;
            guesses.emplace_back(temp);
            total_guesses += 1;
        }
    }
}

void *thread_func(void *arg)
{
    ThreadArg *t_arg = (ThreadArg *)arg;
    for (int i = t_arg->start; i < t_arg->end; ++i)
    {
        t_arg->results->emplace_back(t_arg->guess_prefix + t_arg->a->ordered_values[i]);
    }
    return nullptr;
}

// Generate的pthread并行版本
void PriorityQueue::Generate_pthread(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        if (pt.content[0].type == 3)
            a = &m.symbols[m.FindSymbol(pt.content[0])];

        // pthread并行化
        int n = pt.max_indices[0];
        //

        // guesses.reserve(guesses.size() + n);

        //
        int num_threads = 4; // 可根据实际CPU核数调整
        std::vector<std::vector<std::string>> thread_results(num_threads);
        std::vector<pthread_t> threads(num_threads);
        std::vector<ThreadArg> args(num_threads);

        int chunk = n / num_threads;
        for (int t = 0; t < num_threads; ++t)
        {
            args[t].a = a;
            args[t].guess_prefix = "";
            args[t].start = t * chunk;
            args[t].end = (t == num_threads - 1) ? n : (t + 1) * chunk;
            args[t].results = &thread_results[t];
            pthread_create(&threads[t], nullptr, thread_func, &args[t]);
        }
        for (int t = 0; t < num_threads; ++t)
        {
            pthread_join(threads[t], nullptr);
            guesses.insert(guesses.end(), thread_results[t].begin(), thread_results[t].end());
            total_guesses += thread_results[t].size();
        }
    }
    else
    {
        std::string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 2)
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 3)
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
                break;
        }

        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        if (pt.content[pt.content.size() - 1].type == 2)
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        if (pt.content[pt.content.size() - 1].type == 3)
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];

        // pthread并行化
        int n = pt.max_indices[pt.content.size() - 1];
        // guesses.reserve(guesses.size() + n);
        int num_threads = 4; // 可根据实际CPU核数调整
        std::vector<std::vector<std::string>> thread_results(num_threads);
        std::vector<pthread_t> threads(num_threads);
        std::vector<ThreadArg> args(num_threads);

        int chunk = n / num_threads;
        for (int t = 0; t < num_threads; ++t)
        {
            args[t].a = a;
            args[t].guess_prefix = guess;
            args[t].start = t * chunk;
            args[t].end = (t == num_threads - 1) ? n : (t + 1) * chunk;
            args[t].results = &thread_results[t];
            pthread_create(&threads[t], nullptr, thread_func, &args[t]);
        }
        // 计算结束
        for (int t = 0; t < num_threads; ++t)
        {
            pthread_join(threads[t], nullptr);
            guesses.insert(guesses.end(), thread_results[t].begin(), thread_results[t].end());
            // size_t old_size = guesses.size();
            // guesses.resize(old_size + thread_results[t].size());
            // std::move(thread_results[t].begin(), thread_results[t].end(),
            //          guesses.begin() + old_size);
            total_guesses += thread_results[t].size();
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
void *permanent_thread_func(void *arg)
{
    PermanentThreadArg *t_arg = (PermanentThreadArg *)arg;

    // t_arg->total_work_time = 0.0;
    // t_arg->tasks_processed = 0;

    while (t_arg->running)
    {
        // 等待工作
        pthread_mutex_lock(t_arg->mutex);
        while (!t_arg->has_work && t_arg->running)
        {
            pthread_cond_wait(t_arg->cond, t_arg->mutex);
        }
        // 检查退出信号
        if (!t_arg->running)
        {
            pthread_mutex_unlock(t_arg->mutex);
            break;
        }
        pthread_mutex_unlock(t_arg->mutex);
        // t_arg->start_time = std::chrono::high_resolution_clock::now();
        //  执行工作
        if (t_arg->guesses_ptr != nullptr)
        {
            // 直接写入模式
            size_t pos = t_arg->output_start;
            for (int i = t_arg->start; i < t_arg->end; ++i)
            {
                //(*(t_arg->guesses_ptr))[pos].reserve(32);
                (*(t_arg->guesses_ptr))[pos] = t_arg->prefix;
                (*(t_arg->guesses_ptr))[pos++].append(t_arg->a->ordered_values[i]);
            }
        }
        // t_arg->end_time = std::chrono::high_resolution_clock::now();
        // t_arg->work_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        //                            t_arg->end_time - t_arg->start_time)
        //                            .count();
        // t_arg->total_work_time += t_arg->work_duration;
        // t_arg->tasks_processed += (t_arg->end - t_arg->start);

        // 标记完成
        pthread_mutex_lock(t_arg->mutex);
        t_arg->has_work = false;
        (*t_arg->completed_threads)++;
        // 添加这一行 - 通知主线程
        pthread_cond_broadcast(t_arg->cond);
        pthread_mutex_unlock(t_arg->mutex);
    }
    return nullptr;
}
void PriorityQueue::init_permanent_threads(int n)
{
    num_threads = n;
    completed_threads = n; // 初始状态为全部完成
    pthread_mutex_init(&thread_mutex, nullptr);
    pthread_cond_init(&thread_cond, nullptr);
    // 创建线程
    perm_threads.resize(n);
    for (int i = 0; i < n; ++i)
    {
        perm_threads[i].thread_id = i;
        perm_threads[i].running = true;
        perm_threads[i].has_work = false;
        perm_threads[i].mutex = &thread_mutex;
        perm_threads[i].cond = &thread_cond;
        perm_threads[i].completed_threads = &completed_threads;
        perm_threads[i].results = new std::vector<std::string>();
        pthread_create(&perm_threads[i].thread, nullptr,
                       permanent_thread_func, &perm_threads[i]);
    }
}
void PriorityQueue::cleanup_permanent_threads()
{
    // 发送停止信号
    pthread_mutex_lock(&thread_mutex);
    for (auto &t : perm_threads)
    {
        t.running = false;
    }
    pthread_cond_broadcast(&thread_cond);
    pthread_mutex_unlock(&thread_mutex);
    // 等待所有线程结束
    for (auto &t : perm_threads)
    {
        pthread_join(t.thread, nullptr);
        delete t.results;
    }
    // 清理同步原语
    pthread_mutex_destroy(&thread_mutex);
    pthread_cond_destroy(&thread_cond);
}
void PriorityQueue::Generate_reuse_threads(PT pt)
{
    // static int task_count = 0;
    // if (++task_count % 200 == 0) {
    //     std::cout << "\n==== 线程负载统计 (执行" << task_count << "次) ====" << std::endl;
    //     double total_system_time = 0;
    //     size_t total_tasks = 0;
    //
    //     for (int t = 0; t < num_threads; ++t) {
    //         total_system_time += perm_threads[t].total_work_time;
    //         total_tasks += perm_threads[t].tasks_processed;
    //
    //         std::cout << "线程 " << t
    //                   << ": 处理任务数: " << perm_threads[t].tasks_processed
    //                   << ", 累计工作时间: " << perm_threads[t].total_work_time / 1000.0 << " ms"
    //                   << ", 平均任务时间: " << (perm_threads[t].tasks_processed > 0 ?
    //                      perm_threads[t].total_work_time / perm_threads[t].tasks_processed : 0)
    //                   << " μs/任务" << std::endl;
    //     }
    //
    //     // 计算线程负载均衡指标
    //     double avg_time = total_system_time / num_threads;
    //     double max_deviation = 0;
    //
    //     for (int t = 0; t < num_threads; ++t) {
    //         double deviation = std::abs(perm_threads[t].total_work_time - avg_time) / avg_time * 100.0;
    //         max_deviation = std::max(max_deviation, deviation);
    //     }
    //
    //     std::cout << "负载均衡指标: " << (100.0 - max_deviation) << "% (100%表示完全均衡)" << std::endl;
    //     std::cout << "系统总计: 任务数: " << total_tasks
    //               << ", 总工作时间: " << total_system_time / 1000.0 << " ms" << std::endl;
    //     std::cout << "========================================\n" << std::endl;
    // }
    // 计算PT的概率
    CalProb(pt);
    if (pt.content.size() == 1)
    {
        // 情况1: 只有一个segment
        segment *a;
        if (pt.content[0].type == 1)
            a = &m.letters[m.letter_indices[pt.content[0].length]];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.digit_indices[pt.content[0].length]];
        else
        {
            a = &m.symbols[m.symbol_indices[pt.content[0].length]];
        }
        int n = pt.max_indices[0];
        int old_size = guesses.size();
        guesses.resize(old_size + n);
        // 会有用吗
        if (n < 1150)
        {
            for (int i = 0; i < n; ++i)
            {
                guesses.emplace_back(a->ordered_values[i]);
            }
            total_guesses += n;
            return;
        }
        int chunk = n / num_threads;
        // 分配任务给所有线程
        pthread_mutex_lock(&thread_mutex);
        completed_threads = 0; // 重置计数器
        for (int t = 0; t < num_threads; ++t)
        {
            perm_threads[t].a = a;
            perm_threads[t].prefix = "";
            perm_threads[t].start = t * chunk;
            perm_threads[t].end = (t == num_threads - 1) ? n : (t + 1) * chunk;
            perm_threads[t].output_start = old_size + t * chunk;
            perm_threads[t].guesses_ptr = &guesses;
            perm_threads[t].has_work = true;
        }
        // 唤醒所有线程
        pthread_cond_broadcast(&thread_cond);
        // 等待
        while (completed_threads < num_threads)
        {
            pthread_cond_wait(&thread_cond, &thread_mutex);
        }
        pthread_mutex_unlock(&thread_mutex);

        // 收集结果
        // for (int t = 0; t < num_threads; ++t)
        // {
        //     std::move(perm_threads[t].results->begin(),
        //               perm_threads[t].results->end(),
        //               std::back_inserter(guesses));
        //     // guesses.insert(guesses.end(),
        //     //                perm_threads[t].results->begin(),
        //     //                perm_threads[t].results->end());
        //     //total_guesses += perm_threads[t].results->size();
        // }
        total_guesses += n;
    }
    else
    {
        // 情况2: 多个segment
        std::string guess;
        int seg_idx = 0;
        // 连接除最后一个segment外的所有segment值
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
                guess += m.letters[m.letter_indices[pt.content[seg_idx].length]].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                guess += m.digits[m.digit_indices[pt.content[seg_idx].length]].ordered_values[idx];
            else
                guess += m.symbols[m.symbol_indices[pt.content[seg_idx].length]].ordered_values[idx];
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
                break;
        }
        // 获取最后一个segment
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
            a = &m.letters[m.letter_indices[pt.content[pt.content.size() - 1].length]];
        else if (pt.content[pt.content.size() - 1].type == 2)
            a = &m.digits[m.digit_indices[pt.content[pt.content.size() - 1].length]];
        else
            a = &m.symbols[m.symbol_indices[pt.content[pt.content.size() - 1].length]];
        int n = pt.max_indices[pt.content.size() - 1];
        size_t old_size = guesses.size();
        guesses.resize(old_size + n);
        if (n < 1150)
        {
            for (int i = 0; i < n; ++i)
            {
                guesses.emplace_back(guess);
                guesses.back() += a->ordered_values[i];
            }
            total_guesses += n;
            return;
        }
        int chunk = n / num_threads;

        // 分配任务给所有线程
        pthread_mutex_lock(&thread_mutex);
        completed_threads = 0; // 重置计数器

        for (int t = 0; t < num_threads; ++t)
        {
            perm_threads[t].a = a;
            perm_threads[t].prefix = guess;
            perm_threads[t].start = t * chunk;
            perm_threads[t].end = (t == num_threads - 1) ? n : (t + 1) * chunk;
            perm_threads[t].output_start = old_size + t * chunk;
            perm_threads[t].guesses_ptr = &guesses;
            perm_threads[t].has_work = true;
        }
        // 唤醒所有线程开始工作
        pthread_cond_broadcast(&thread_cond);
        // 等待所有线程完成
        while (completed_threads < num_threads)
        {
            pthread_cond_wait(&thread_cond, &thread_mutex);
        }
        pthread_mutex_unlock(&thread_mutex);
        // 收集结果
        // guesses.reserve(guesses.size() + n);
        // for (int t = 0; t < num_threads; ++t)
        // {
        //     std::move(perm_threads[t].results->begin(),
        //               perm_threads[t].results->end(),
        //               std::back_inserter(guesses));
        //     // 此处确实加速
        //     //  guesses.insert(guesses.end(),
        //     //                 perm_threads[t].results->begin(),
        //     //                 perm_threads[t].results->end());
        //     //  total_guesses += perm_threads[t].results->size();
        // }
        total_guesses += n;
    }
}

// void* permanent_thread_func(void* arg) {
//     PermanentThreadArg* t_arg = (PermanentThreadArg*)arg;//
//     while(t_arg->running) {
//         // 使用原子变量进行自旋等待任务，无需加锁
//         while(!t_arg->has_work.load(std::memory_order_acquire) && t_arg->running) {
//             // 短暂暂停以减少CPU使用
//             usleep(10);
//         }//
//         // 检查退出信号
//         if(!t_arg->running) {
//             break;
//         }//
//         // 执行工作
//         int resultsize = t_arg->end - t_arg->start;
//         t_arg->results->clear();
//         t_arg->results->reserve(resultsize);//
//         for(int i = t_arg->start; i < t_arg->end; ++i) {
//             t_arg->results->emplace_back(t_arg->prefix + t_arg->a->ordered_values[i]);
//         }//
//         // 使用原子操作标记完成
//         t_arg->has_work.store(false, std::memory_order_release);
//         t_arg->work_done.store(true, std::memory_order_release);
//     }//
//     return nullptr;
// }//
// void PriorityQueue::init_permanent_threads(int n) {
//     num_threads = n;
//     completed_threads.store(0);//
//     // 预留空间
//     perm_threads.reserve(n);//
//     // 动态创建线程参数对象
//     for(int i = 0; i < n; ++i) {
//         // 使用new分配对象
//         PermanentThreadArg* arg = new PermanentThreadArg();
//         arg->thread_id = i;
//         arg->running = true;
//         arg->has_work.store(false, std::memory_order_relaxed);
//         arg->work_done.store(true, std::memory_order_relaxed);
//         arg->results = new std::vector<std::string>();//
//         // 添加指针到向量
//         perm_threads.push_back(arg);//
//         // 创建线程
//         pthread_create(&arg->thread, nullptr, permanent_thread_func, arg);
//     }
// }//
// void PriorityQueue::cleanup_permanent_threads() {
//     // 发送停止信号
//     for(auto t : perm_threads) {
//         t->running = false;
//         t->has_work.store(true, std::memory_order_release); // 唤醒等待中的线程
//     }//
//     // 等待所有线程结束
//     for(auto t : perm_threads) {
//         pthread_join(t->thread, nullptr);
//         delete t->results;
//         delete t;  // 释放动态分配的线程参数对象
//     }//
//     // 清空容器
//     perm_threads.clear();
// }//
// void PriorityQueue::Generate_reuse_threads(PT pt) {
//     // 计算PT的概率
//     CalProb(pt);//
//     // 根据有几个segment选择不同的处理分支
//     if(pt.content.size() == 1) {
//         // 情况1: 只有一个segment
//         segment* a;
//         if(pt.content[0].type == 1)
//             a = &m.letters[m.FindLetter(pt.content[0])];
//         else if(pt.content[0].type == 2)
//             a = &m.digits[m.FindDigit(pt.content[0])];
//         else
//             a = &m.symbols[m.FindSymbol(pt.content[0])];//
//         int n = pt.max_indices[0];
//         int chunk = n / num_threads;//
//         // 预分配空间，避免后续多次扩容
//         size_t total_size = 0;
//         for(int t = 0; t < num_threads; ++t) {
//             perm_threads[t]->results->clear();
//             int local_chunk = (t == num_threads - 1) ? (n - t * chunk) : chunk;
//             perm_threads[t]->results->reserve(local_chunk);
//             total_size += local_chunk;
//         }
//         guesses.reserve(guesses.size() + total_size);//
//         // 重置完成计数
//         completed_threads.store(0, std::memory_order_relaxed);//
//         // 分配任务给所有线程
//         for(int t = 0; t < num_threads; ++t) {
//             // 设置任务参数
//             perm_threads[t]->a = a;
//             perm_threads[t]->prefix = "";
//             perm_threads[t]->start = t * chunk;
//             perm_threads[t]->end = (t == num_threads - 1) ? n : (t + 1) * chunk;
//             perm_threads[t]->work_done.store(false, std::memory_order_relaxed);//
//             // 启动任务
//             perm_threads[t]->has_work.store(true, std::memory_order_release);
//         }//
//         // 等待所有线程完成 - 使用自旋等待
//         for(int t = 0; t < num_threads; ++t) {
//             while(!perm_threads[t]->work_done.load(std::memory_order_acquire)) {
//                 // 短暂暂停，避免过度占用CPU
//                 usleep(10);
//             }
//         }//
//         // 收集结果 - 使用移动语义优化
//         for(int t = 0; t < num_threads; ++t) {
//             size_t oldSize = guesses.size();
//             size_t newElements = perm_threads[t]->results->size();//
//             // 扩展并一次性移动元素
//             guesses.resize(oldSize + newElements);
//             std::move(perm_threads[t]->results->begin(),
//                      perm_threads[t]->results->end(),
//                      guesses.begin() + oldSize);//
//             total_guesses += newElements;
//         }
//     }
//     else {
//         // 情况2: 多个segment
//         std::string guess;
//         int seg_idx = 0;//
//         // 连接除最后一个segment外的所有segment值
//         for(int idx : pt.curr_indices) {
//             if(pt.content[seg_idx].type == 1)
//                 guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
//             else if(pt.content[seg_idx].type == 2)
//                 guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
//             else
//                 guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];//
//             seg_idx += 1;
//             if(seg_idx == pt.content.size() - 1)
//                 break;
//         }//
//         // 获取最后一个segment
//         segment* a;
//         if(pt.content[pt.content.size() - 1].type == 1)
//             a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
//         else if(pt.content[pt.content.size() - 1].type == 2)
//             a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
//         else
//             a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];//
//         int n = pt.max_indices[pt.content.size() - 1];
//         int chunk = n / num_threads;//
//         // 同样预分配空间
//         size_t total_size = 0;
//         for(int t = 0; t < num_threads; ++t) {
//             perm_threads[t]->results->clear();
//             int local_chunk = (t == num_threads - 1) ? (n - t * chunk) : chunk;
//             perm_threads[t]->results->reserve(local_chunk);
//             total_size += local_chunk;
//         }
//         guesses.reserve(guesses.size() + total_size);//
//         // 分配任务给所有线程
//         for(int t = 0; t < num_threads; ++t) {
//             perm_threads[t]->a = a;
//             perm_threads[t]->prefix = guess;
//             perm_threads[t]->start = t * chunk;
//             perm_threads[t]->end = (t == num_threads - 1) ? n : (t + 1) * chunk;
//             perm_threads[t]->work_done.store(false, std::memory_order_relaxed);//
//             // 启动任务
//             perm_threads[t]->has_work.store(true, std::memory_order_release);
//         }//
//         // 等待所有线程完成
//         for(int t = 0; t < num_threads; ++t) {
//             while(!perm_threads[t]->work_done.load(std::memory_order_acquire)) {
//                 usleep(5);
//             }
//         }//
//         // 收集结果
//         for(int t = 0; t < num_threads; ++t) {
//             size_t oldSize = guesses.size();
//             size_t newElements = perm_threads[t]->results->size();//
//             guesses.resize(oldSize + newElements);
//             std::move(perm_threads[t]->results->begin(),
//                      perm_threads[t]->results->end(),
//                      guesses.begin() + oldSize);//
//             total_guesses += newElements;
//         }
//     }
// }

// OpenMP
void PriorityQueue::Generate_OpenMP(PT pt)
{
    // 计算PT的概率，这里主要是给PT的概率进行初始化
    CalProb(pt);
    omp_set_num_threads(4);
    int thread_count = 4;
    // 对于只有一个segment的PT，直接遍历生成其中的所有value即可
    if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
            a = &m.letters[m.letter_indices[pt.content[0].length]];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.digit_indices[pt.content[0].length]];
        else
        {
            a = &m.symbols[m.symbol_indices[pt.content[0].length]];
        }
        int n = pt.max_indices[0];
        size_t old_size = guesses.size();
        // 预分配空间以避免动态扩容开销
        guesses.resize(old_size + n);

        if (n < 1150)
        {
            // 小任务
            for (int i = 0; i < n; i++)
            {
                guesses[old_size + i] = a->ordered_values[i];
            }
        }
        else
        {
            // 预分配线程本地存储
            int chunk = n / thread_count;

#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int start = tid * chunk;
                int end = (tid == thread_count - 1) ? n : (tid + 1) * chunk;

                // 直接写入guesses数组
                for (int i = start; i < end; i++)
                {
                    guesses[old_size + i] = a->ordered_values[i];
                }
            }
        }
        total_guesses += n;
    }
    else
    {
        string guess;
        int seg_idx = 0;
        // 这个for循环的作用：给当前PT的所有segment赋予实际的值（最后一个segment除外）
        // segment值根据curr_indices中对应的值加以确定
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
                guess += m.letters[m.letter_indices[pt.content[seg_idx].length]].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                guess += m.digits[m.digit_indices[pt.content[seg_idx].length]].ordered_values[idx];
            else
                guess += m.symbols[m.symbol_indices[pt.content[seg_idx].length]].ordered_values[idx];
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
            {
                break;
            }
        }

        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
            a = &m.letters[m.letter_indices[pt.content[pt.content.size() - 1].length]];
        else if (pt.content[pt.content.size() - 1].type == 2)
            a = &m.digits[m.digit_indices[pt.content[pt.content.size() - 1].length]];
        else
            a = &m.symbols[m.symbol_indices[pt.content[pt.content.size() - 1].length]];

        int n = pt.max_indices[pt.content.size() - 1];
        // 预分配空间以避免动态扩容开销
        int old_size = guesses.size();
        guesses.resize(old_size + n);

        if (n < 1150)
        {
            // guesses.reserve(guesses.size() + n);
            for (int i = 0; i < n; i++)
            {
                guesses[old_size + i].reserve(2 * guess.size());
                // 预分配是否有效：不确定，需要单独测试
                guesses[old_size + i] += guess;
                guesses[old_size + i] += a->ordered_values[i];
                // guesses.emplace_back(guess);
                // guesses.back().append(a->ordered_values[i]);
            }
            // total_guesses += n;
        }
        else
        {
            // 预分配空间避免竞争
            // int thread_count = omp_get_max_threads();
            // std::vector<std::vector<std::string>> all_results(thread_count);
            int chunk = n / thread_count;

#pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int start = tid * chunk;
                int end = (tid == thread_count - 1) ? n : (tid + 1) * chunk;

                // 直接写入guesses数组
                for (int i = start; i < end; i++)
                {
                    // 仍然需要测试reserve的性能：单独写
                    guesses[old_size + i].reserve(2 * guess.size());
                    guesses[old_size + i] += guess;
                    guesses[old_size + i] += a->ordered_values[i];
                }
            }
        }
        total_guesses += n;
    }
}



void PriorityQueue::Generate_OpenMp_Simple(PT pt)
{
    // 计算PT的概率
    CalProb(pt);
    
    // 对于只有一个segment的PT
    if (pt.content.size() == 1)
    {
        // 获取segment指针
        segment *a;
        if (pt.content[0].type == 1)
            a = &m.letters[m.letter_indices[pt.content[0].length]];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.digit_indices[pt.content[0].length]];
        else
            a = &m.symbols[m.symbol_indices[pt.content[0].length]];
        
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
        // 构建前缀
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1)
                guess += m.letters[m.letter_indices[pt.content[seg_idx].length]].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                guess += m.digits[m.digit_indices[pt.content[seg_idx].length]].ordered_values[idx];
            else
                guess += m.symbols[m.symbol_indices[pt.content[seg_idx].length]].ordered_values[idx];
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
                break;
        }
        
        // 获取最后一个segment
        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
            a = &m.letters[m.letter_indices[pt.content[pt.content.size() - 1].length]];
        else if (pt.content[pt.content.size() - 1].type == 2)
            a = &m.digits[m.digit_indices[pt.content[pt.content.size() - 1].length]];
        else
            a = &m.symbols[m.symbol_indices[pt.content[pt.content.size() - 1].length]];
        
        int n = pt.max_indices[pt.content.size() - 1];
        int old_size = guesses.size();
        guesses.resize(old_size + n);
        
        if (n < 1150) {
            // 小任务串行执行
            for (int i = 0; i < n; i++) {
                guesses[old_size + i] = guess + a->ordered_values[i];
            }
        } else {
            // 简化的OpenMP并行for循环
            #pragma omp parallel for num_threads(4) schedule(dynamic, 1000)
            for (int i = 0; i < n; i++) {
                guesses[old_size + i] = guess;
                guesses[old_size + i] += a->ordered_values[i];
            }
        }
        total_guesses += n;
    }
}


void PriorityQueue::Generate_Omp(PT pt) {
    // 计算PT的概率
    CalProb(pt);

    // 对于只有一个segment的PT
     if (pt.content.size() == 1)
    {
        // 指向最后一个segment的指针，这个指针实际指向模型中的统计数据
        segment *a;
        // 在模型中定位到这个segment
        if (pt.content[0].type == 1)
            a = &m.letters[m.letter_indices[pt.content[0].length]];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.digit_indices[pt.content[0].length]];
        else
        {
            a = &m.symbols[m.symbol_indices[pt.content[0].length]];
        }
        
        #pragma omp parallel
        {
            vector<string> local_guesses;
            int local_total = 0;
            const int max_idx = pt.max_indices[0];
            
            // 预分配空间
            local_guesses.resize(max_idx);
            
            #pragma omp for schedule(dynamic, 1000)
            for (int i = 0; i < max_idx; i++) {
                local_guesses[i] = a->ordered_values[i];
                local_total = max_idx; 
            }
            
            #pragma omp critical
            {
                guesses.insert(guesses.end(), local_guesses.begin(), local_guesses.begin() + local_total);
                total_guesses += local_total;
            }
        }
    }
    else {
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1)
                guess += m.letters[m.letter_indices[pt.content[seg_idx].length]].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                guess += m.digits[m.digit_indices[pt.content[seg_idx].length]].ordered_values[idx];
            else
                guess += m.symbols[m.symbol_indices[pt.content[seg_idx].length]].ordered_values[idx];
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
                break;
        }

        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
            a = &m.letters[m.letter_indices[pt.content[pt.content.size() - 1].length]];
        else if (pt.content[pt.content.size() - 1].type == 2)
            a = &m.digits[m.digit_indices[pt.content[pt.content.size() - 1].length]];
        else
            a = &m.symbols[m.symbol_indices[pt.content[pt.content.size() - 1].length]];

        #pragma omp parallel
        {
            vector<string> local_guesses;
            int local_total = 0;
            const int max_idx = pt.max_indices[pt.content.size() - 1];

            local_guesses.resize(max_idx);
            
            #pragma omp for schedule(dynamic, 1000)
            for (int i = 0; i < max_idx; i++) {
                local_guesses[i] = guess ;
                local_guesses[i] += a->ordered_values[i];
                local_total = max_idx; 
            }
            
            #pragma omp critical
            {
                guesses.insert(guesses.end(), local_guesses.begin(), local_guesses.begin() + local_total);
                total_guesses += local_total;
            }
        }
    }
}
