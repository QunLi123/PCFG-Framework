#pragma once

#include <iostream>
#include <string>
#include <cstring>
#include<immintrin.h>
#include<vector>

using namespace std;

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef unsigned int bit32;

// MD5的一系列参数。参数是固定的，其实你不需要看懂这些
#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21

/**
 * @Basic MD5 functions.
 *
 * @param there bit32.
 *
 * @return one bit32.
 */
 // 定义了一系列MD5中的具体函数
 // 这四个计算函数是需要你进行SIMD并行化的
 // 可以看到，FGHI四个函数都涉及一系列位运算，在数据上是对齐的，非常容易实现SIMD的并行化

#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

/**
 * @Rotate Left.
 *
 * @param {num} the raw number.
 *
 * @param {n} rotate left n.
 *
 * @return the number after rotated left.
 */
 // 定义了一系列MD5中的具体函数
 // 这五个计算函数（ROTATELEFT/FF/GG/HH/II）和之前的FGHI一样，都是需要你进行SIMD并行化的
 // 但是你需要注意的是#define的功能及其效果，可以发现这里的FGHI是没有返回值的，为什么呢？你可以查询#define的含义和用法
#define ROTATELEFT(num, n) (((num) << (n)) | ((num) >> (32-(n))))

#define FF(a, b, c, d, x, s, ac) { \
  (a) += F ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

#define GG(a, b, c, d, x, s, ac) { \
  (a) += G ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}
#define HH(a, b, c, d, x, s, ac) { \
  (a) += H ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}
#define II(a, b, c, d, x, s, ac) { \
  (a) += I ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

void MD5Hash(string input, bit32* state);


// SIMD版本的FGHI函数
inline __m256i F_SIMD(__m256i x, __m256i y, __m256i z) {
    return _mm256_or_si256(_mm256_and_si256(x, y), _mm256_andnot_si256(x, z));
}

inline __m256i G_SIMD(__m256i x, __m256i y, __m256i z) {
    return _mm256_or_si256(_mm256_and_si256(x, z), _mm256_andnot_si256(z, y));
}

inline __m256i H_SIMD(__m256i x, __m256i y, __m256i z) {
    return _mm256_xor_si256(_mm256_xor_si256(x, y), z);
}

inline __m256i I_SIMD(__m256i x, __m256i y, __m256i z) {
    return _mm256_xor_si256(y, _mm256_or_si256(x, _mm256_andnot_si256(z, _mm256_set1_epi32(0xffffffff))));
}

// SIMD循环左移
inline __m256i ROTATELEFT_SIMD(__m256i num, int n) {
    return _mm256_or_si256(_mm256_slli_epi32(num, n), _mm256_srli_epi32(num, 32 - n));
}

// SIMD化FF、GG、HH、II宏
#define FF_SIMD(a, b, c, d, x, s, ac) { \
  a = _mm256_add_epi32(a, _mm256_add_epi32(F_SIMD(b,c,d), _mm256_add_epi32(x, _mm256_set1_epi32(ac)))); \
  a = ROTATELEFT_SIMD(a, s); \
  a = _mm256_add_epi32(a, b); \
}

#define GG_SIMD(a, b, c, d, x, s, ac) { \
  a = _mm256_add_epi32(a, _mm256_add_epi32(G_SIMD(b,c,d), _mm256_add_epi32(x, _mm256_set1_epi32(ac)))); \
  a = ROTATELEFT_SIMD(a, s); \
  a = _mm256_add_epi32(a, b); \
}

#define HH_SIMD(a, b, c, d, x, s, ac) { \
  a = _mm256_add_epi32(a, _mm256_add_epi32(H_SIMD(b,c,d), _mm256_add_epi32(x, _mm256_set1_epi32(ac)))); \
  a = ROTATELEFT_SIMD(a, s); \
  a = _mm256_add_epi32(a, b); \
}

#define II_SIMD(a, b, c, d, x, s, ac) { \
  a = _mm256_add_epi32(a, _mm256_add_epi32(I_SIMD(b,c,d), _mm256_add_epi32(x, _mm256_set1_epi32(ac)))); \
  a = ROTATELEFT_SIMD(a, s); \
  a = _mm256_add_epi32(a, b); \
}



void MD5Hash_SIMD8(const string inputs[8], bit32 states[8][4]);