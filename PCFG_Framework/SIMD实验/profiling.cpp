// test_md5.cpp
#include "md5.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
using namespace std;
using namespace chrono;
//g++ profiling.cpp md5.cpp -o profiling -O2
// 测试参数
const int NUM_PASSWORDS = 10000000; // 调整为合适的数量
const int PASSWORD_LENGTH = 10;    // 平均密码长度

// 生成测试数据
vector<string> generate_test_passwords(int count, int length)
{
    vector<string> passwords;
    passwords.reserve(count);
    for (int i = 0; i < count; i++)
    {
        string pw;
        pw.reserve(length);
        for (int j = 0; j < length; j++)
        {
            pw += 'a' + ((i + j) % 26);
        }
        passwords.push_back(pw);
    }
    return passwords;
}

// 测试串行MD5
void test_serial_md5()
{
    auto passwords = generate_test_passwords(NUM_PASSWORDS, PASSWORD_LENGTH);
    bit32 state[4];

    auto start = high_resolution_clock::now();
    for (const auto &pw : passwords)
    {
        MD5Hash(pw, state);
    }
    auto end = high_resolution_clock::now();

    cout << "串行MD5处理" << NUM_PASSWORDS << "个密码用时: "
         << duration_cast<milliseconds>(end - start).count() << "毫秒" << endl;
}

// 测试四路SIMD MD5
void test_simd4_md5()
{
    auto passwords = generate_test_passwords(NUM_PASSWORDS, PASSWORD_LENGTH);

    auto start = high_resolution_clock::now();
    alignas(16) bit32 states[4][4];
    string strs[4];
    for (int i = 0; i < passwords.size(); i += 4)
    {
        strs[0] = passwords[i];
        strs[1] = i + 1 < passwords.size() ? passwords[i + 1] : passwords[i];
        strs[2] = i + 2 < passwords.size() ? passwords[i + 2] : passwords[i];
        strs[3] = i + 3 < passwords.size() ? passwords[i + 3] : passwords[i];
        MD5Hash_SIMD4(strs, states);
    }
    auto end = high_resolution_clock::now();

    cout << "四路SIMD MD5处理" << NUM_PASSWORDS << "个密码用时: "
         << duration_cast<milliseconds>(end - start).count() << "毫秒" << endl;
}

int main(int argc, char *argv[])
{
     if (argc < 2)
     {
         cout << "使用方法: " << argv[0] << " [serial|simd4|both]" << endl;
         return 1;
     }

     string mode = argv[1];
     if (mode == "serial" || mode == "both")
     {
         test_serial_md5();
     }
     if (mode == "simd4" || mode == "both")
     {
         test_simd4_md5();
     }
    return 0;
}