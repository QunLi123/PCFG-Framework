#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
using namespace std;
using namespace chrono;

// 编译指令如下：
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o test.exe


// 通过这个函数，你可以验证你实现的SIMD哈希函数的正确性
int main()
{
    
    bit32 state[4];
    MD5Hash("bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva", state);
    for (int i1 = 0; i1 < 4; i1 += 1)
    {
        cout << std::setw(8) << std::setfill('0') << hex << state[i1];
    }
    cout << endl;
    // vector<string>input(1);input[0]="bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva";
    // int lengths[1];lengths[0]=input[0].length();
    // bit32 states[1][4];
    // MD5Hash_SIMD(input, lengths, 1, states);
    // for (int i1 = 0; i1 < 4; i1 += 1)
    // {
    //     cout << std::setw(8) << std::setfill('0') << hex << states[0][i1];
    // }

    cout<<endl;
    cout << endl;
    string longstr="bvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdvabvaisdbjasdkafkasdfnavkjnakdjfejfanjsdnfkajdfkajdfjkwanfdjaknsvjkanbjbjadfajwefajksdfakdnsvjadfasjdva";
    string strbs[4];
    strbs[0] = "abcd";
    strbs[1] = "efgh";
    strbs[2] = "ijkl";
    strbs[3] = "longstr";
    int lengths[4];
    lengths[0] = 4;
    lengths[1] = 4;
    lengths[2] = 4;
    lengths[3] = 7;
    bit32 states[4][4];
    MD5Hash_SIMD4(strbs, states);
    cout << "并行结果：" << endl;
    for (int i = 0; i < 4; i++)
    {
        for (int i1 = 0; i1 < 4; i1 += 1)
        {

            cout << std::setw(8) << std::setfill('0') << hex << states[i][i1];
        }
        cout << endl;
    }


    cout << "串行结果：" << endl;
    for(int i = 0; i < 4; i++)
    {
        bit32 s[4];
        MD5Hash(strbs[i], s);
        
        for (int i1 = 0; i1 < 4; i1 += 1)
        {
            cout << std::setw(8) << std::setfill('0') << hex << s[i1];
        }
        cout << endl;
    }
}