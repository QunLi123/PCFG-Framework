#pragma once

#include <iostream>
#include <string>
#include <cstring>
#include<immintrin.h>
#include<vector>

using namespace std;

// ������Byte������ʹ��
typedef unsigned char Byte;
// ������32����
typedef unsigned int bit32;

// MD5��һϵ�в����������ǹ̶��ģ���ʵ�㲻��Ҫ������Щ
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
 // ������һϵ��MD5�еľ��庯��
 // ���ĸ����㺯������Ҫ�����SIMD���л���
 // ���Կ�����FGHI�ĸ��������漰һϵ��λ���㣬���������Ƕ���ģ��ǳ�����ʵ��SIMD�Ĳ��л�

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
 // ������һϵ��MD5�еľ��庯��
 // ��������㺯����ROTATELEFT/FF/GG/HH/II����֮ǰ��FGHIһ����������Ҫ�����SIMD���л���
 // ��������Ҫע�����#define�Ĺ��ܼ���Ч�������Է��������FGHI��û�з���ֵ�ģ�Ϊʲô�أ�����Բ�ѯ#define�ĺ�����÷�
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


// SIMD�汾��FGHI����
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

// SIMDѭ������
inline __m256i ROTATELEFT_SIMD(__m256i num, int n) {
    return _mm256_or_si256(_mm256_slli_epi32(num, n), _mm256_srli_epi32(num, 32 - n));
}

// SIMD��FF��GG��HH��II��
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