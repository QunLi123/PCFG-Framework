#include <iostream>
#include <string>
#include <cstring>
#include <vector>

#ifdef _MSC_VER
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
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

void MD5Hash(string input, bit32 *state);


#define F_SIMD(x,y,z) vbslq_u32(x, y, z)
#define G_SIMD(x,y,z) vbslq_u32(z, x, y)
#define H_SIMD(x,y,z) veorq_u32(veorq_u32(x, y), z)
#define I_SIMD(x,y,z) veorq_u32(y, vorrq_u32(x, vmvnq_u32(z)))
#define ROTATELEFT_SIMD(x, n) vorrq_u32(vshlq_n_u32(x, n), vshrq_n_u32(x, 32 - n))

// #define FF(a, b, c, d, x, s, ac) { \
//   (a) += F ((b), (c), (d)) + (x) + ac; \
//   (a) = ROTATELEFT ((a), (s)); \
//   (a) += (b); \
// }
// NEON版本的F/G/H/I轮函数宏定义
#define FF_SIMD(a, b, c, d, x, s, ac) { \
  a= vaddq_u32(a, vaddq_u32(F_SIMD(b,c,d),vaddq_u32(x,vdupq_n_u32(ac)))); \
  a= ROTATELEFT_SIMD(a, s); \
  a= vaddq_u32(a, b); \
}
// #define GG(a, b, c, d, x, s, ac) { \
//   (a) += G ((b), (c), (d)) + (x) + ac; \
//   (a) = ROTATELEFT ((a), (s)); \
//   (a) += (b); \
// }
#define GG_SIMD(a, b, c, d, x, s, ac) { \
  a= vaddq_u32(a, vaddq_u32(G_SIMD(b,c,d), vaddq_u32(x,vdupq_n_u32(ac)))); \
  a= ROTATELEFT_SIMD(a, s); \
  a= vaddq_u32(a, b); \
}

#define HH_SIMD(a, b, c, d, x, s, ac) { \
  a= vaddq_u32(a, vaddq_u32(H_SIMD(b,c,d), vaddq_u32(x,vdupq_n_u32(ac)))); \
  a= ROTATELEFT_SIMD(a, s); \
  a= vaddq_u32(a, b); \
}

#define II_SIMD(a, b, c, d, x, s, ac) { \
  a= vaddq_u32(a, vaddq_u32(I_SIMD(b,c,d), vaddq_u32(x,vdupq_n_u32(ac)))); \
  a= ROTATELEFT_SIMD(a, s); \
  a= vaddq_u32(a, b); \
}


// static const uint32_t md5_constants[] = {
//   0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
//   0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
//   0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
//   0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
//   0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
//   0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
//   0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
//   0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
// };

// 将内联函数改为宏定义
#define ROTATELEFT_SIMD_7(x) vorrq_u32(vshlq_n_u32(x, 7), vshrq_n_u32(x, 25))
#define ROTATELEFT_SIMD_12(x) vorrq_u32(vshlq_n_u32(x, 12), vshrq_n_u32(x, 20))
#define ROTATELEFT_SIMD_17(x) vorrq_u32(vshlq_n_u32(x, 17), vshrq_n_u32(x, 15))
#define ROTATELEFT_SIMD_22(x) vorrq_u32(vshlq_n_u32(x, 22), vshrq_n_u32(x, 10))
#define ROTATELEFT_SIMD_5(x) vorrq_u32(vshlq_n_u32(x, 5), vshrq_n_u32(x, 27))
#define ROTATELEFT_SIMD_9(x) vorrq_u32(vshlq_n_u32(x, 9), vshrq_n_u32(x, 23))
#define ROTATELEFT_SIMD_14(x) vorrq_u32(vshlq_n_u32(x, 14), vshrq_n_u32(x, 18))
#define ROTATELEFT_SIMD_20(x) vorrq_u32(vshlq_n_u32(x, 20), vshrq_n_u32(x, 12))
#define ROTATELEFT_SIMD_4(x) vorrq_u32(vshlq_n_u32(x, 4), vshrq_n_u32(x, 28))
#define ROTATELEFT_SIMD_11(x) vorrq_u32(vshlq_n_u32(x, 11), vshrq_n_u32(x, 21))
#define ROTATELEFT_SIMD_16(x) vorrq_u32(vshlq_n_u32(x, 16), vshrq_n_u32(x, 16))
#define ROTATELEFT_SIMD_23(x) vorrq_u32(vshlq_n_u32(x, 23), vshrq_n_u32(x, 9))
#define ROTATELEFT_SIMD_6(x) vorrq_u32(vshlq_n_u32(x, 6), vshrq_n_u32(x, 26))
#define ROTATELEFT_SIMD_10(x) vorrq_u32(vshlq_n_u32(x, 10), vshrq_n_u32(x, 22))
#define ROTATELEFT_SIMD_15(x) vorrq_u32(vshlq_n_u32(x, 15), vshrq_n_u32(x, 17))
#define ROTATELEFT_SIMD_21(x) vorrq_u32(vshlq_n_u32(x, 21), vshrq_n_u32(x, 11))








#define FF_ROUND_FULLY_UNROLLED(a, b, c, d, x_vec) \
  /* 步骤1 */ \
  a = vaddq_u32(a, vaddq_u32(F_SIMD(b, c, d), vaddq_u32(x_vec[0], vdupq_n_u32(0xd76aa478)))); \
  a = ROTATELEFT_SIMD_7(a); \
  a = vaddq_u32(a, b); \
  \
  /* 步骤2 */ \
  d = vaddq_u32(d, vaddq_u32(F_SIMD(a, b, c), vaddq_u32(x_vec[1], vdupq_n_u32(0xe8c7b756)))); \
  d = ROTATELEFT_SIMD_12(d); \
  d = vaddq_u32(d, a); \
  \
  /* 步骤3 */ \
  c = vaddq_u32(c, vaddq_u32(F_SIMD(d, a, b), vaddq_u32(x_vec[2], vdupq_n_u32(0x242070db)))); \
  c = ROTATELEFT_SIMD_17(c); \
  c = vaddq_u32(c, d); \
  \
  /* 步骤4 */ \
  b = vaddq_u32(b, vaddq_u32(F_SIMD(c, d, a), vaddq_u32(x_vec[3], vdupq_n_u32(0xc1bdceee)))); \
  b = ROTATELEFT_SIMD_22(b); \
  b = vaddq_u32(b, c); \
  \
  /* 步骤5 */ \
  a = vaddq_u32(a, vaddq_u32(F_SIMD(b, c, d), vaddq_u32(x_vec[4], vdupq_n_u32(0xf57c0faf)))); \
  a = ROTATELEFT_SIMD_7(a); \
  a = vaddq_u32(a, b); \
  \
  /* 步骤6 */ \
  d = vaddq_u32(d, vaddq_u32(F_SIMD(a, b, c), vaddq_u32(x_vec[5], vdupq_n_u32(0x4787c62a)))); \
  d = ROTATELEFT_SIMD_12(d); \
  d = vaddq_u32(d, a); \
  \
  /* 步骤7 */ \
  c = vaddq_u32(c, vaddq_u32(F_SIMD(d, a, b), vaddq_u32(x_vec[6], vdupq_n_u32(0xa8304613)))); \
  c = ROTATELEFT_SIMD_17(c); \
  c = vaddq_u32(c, d); \
  \
  /* 步骤8 */ \
  b = vaddq_u32(b, vaddq_u32(F_SIMD(c, d, a), vaddq_u32(x_vec[7], vdupq_n_u32(0xfd469501)))); \
  b = ROTATELEFT_SIMD_22(b); \
  b = vaddq_u32(b, c); \
  \
  /* 步骤9 */ \
  a = vaddq_u32(a, vaddq_u32(F_SIMD(b, c, d), vaddq_u32(x_vec[8], vdupq_n_u32(0x698098d8)))); \
  a = ROTATELEFT_SIMD_7(a); \
  a = vaddq_u32(a, b); \
  \
  /* 步骤10 */ \
  d = vaddq_u32(d, vaddq_u32(F_SIMD(a, b, c), vaddq_u32(x_vec[9], vdupq_n_u32(0x8b44f7af)))); \
  d = ROTATELEFT_SIMD_12(d); \
  d = vaddq_u32(d, a); \
  \
  /* 步骤11 */ \
  c = vaddq_u32(c, vaddq_u32(F_SIMD(d, a, b), vaddq_u32(x_vec[10], vdupq_n_u32(0xffff5bb1)))); \
  c = ROTATELEFT_SIMD_17(c); \
  c = vaddq_u32(c, d); \
  \
  /* 步骤12 */ \
  b = vaddq_u32(b, vaddq_u32(F_SIMD(c, d, a), vaddq_u32(x_vec[11], vdupq_n_u32(0x895cd7be)))); \
  b = ROTATELEFT_SIMD_22(b); \
  b = vaddq_u32(b, c); \
  \
  /* 步骤13 */ \
  a = vaddq_u32(a, vaddq_u32(F_SIMD(b, c, d), vaddq_u32(x_vec[12], vdupq_n_u32(0x6b901122)))); \
  a = ROTATELEFT_SIMD_7(a); \
  a = vaddq_u32(a, b); \
  \
  /* 步骤14 */ \
  d = vaddq_u32(d, vaddq_u32(F_SIMD(a, b, c), vaddq_u32(x_vec[13], vdupq_n_u32(0xfd987193)))); \
  d = ROTATELEFT_SIMD_12(d); \
  d = vaddq_u32(d, a); \
  \
  /* 步骤15 */ \
  c = vaddq_u32(c, vaddq_u32(F_SIMD(d, a, b), vaddq_u32(x_vec[14], vdupq_n_u32(0xa679438e)))); \
  c = ROTATELEFT_SIMD_17(c); \
  c = vaddq_u32(c, d); \
  \
  /* 步骤16 */ \
  b = vaddq_u32(b, vaddq_u32(F_SIMD(c, d, a), vaddq_u32(x_vec[15], vdupq_n_u32(0x49b40821)))); \
  b = ROTATELEFT_SIMD_22(b); \
  b = vaddq_u32(b, c);

#define GG_ROUND_FULLY_UNROLLED(a, b, c, d, x_vec) \
  /* 步骤1 */ \
  a = vaddq_u32(a, vaddq_u32(G_SIMD(b, c, d), vaddq_u32(x_vec[1], vdupq_n_u32(0xf61e2562)))); \
  a = ROTATELEFT_SIMD_5(a); \
  a = vaddq_u32(a, b); \
  \
  /* 步骤2 */ \
  d = vaddq_u32(d, vaddq_u32(G_SIMD(a, b, c), vaddq_u32(x_vec[6], vdupq_n_u32(0xc040b340)))); \
  d = ROTATELEFT_SIMD_9(d); \
  d = vaddq_u32(d, a); \
  \
  /* 步骤3 */ \
  c = vaddq_u32(c, vaddq_u32(G_SIMD(d, a, b), vaddq_u32(x_vec[11], vdupq_n_u32(0x265e5a51)))); \
  c = ROTATELEFT_SIMD_14(c); \
  c = vaddq_u32(c, d); \
  \
  /* 步骤4 */ \
  b = vaddq_u32(b, vaddq_u32(G_SIMD(c, d, a), vaddq_u32(x_vec[0], vdupq_n_u32(0xe9b6c7aa)))); \
  b = ROTATELEFT_SIMD_20(b); \
  b = vaddq_u32(b, c); \
  \
  /* 步骤5 */ \
  a = vaddq_u32(a, vaddq_u32(G_SIMD(b, c, d), vaddq_u32(x_vec[5], vdupq_n_u32(0xd62f105d)))); \
  a = ROTATELEFT_SIMD_5(a); \
  a = vaddq_u32(a, b); \
  \
  /* 步骤6 */ \
  d = vaddq_u32(d, vaddq_u32(G_SIMD(a, b, c), vaddq_u32(x_vec[10], vdupq_n_u32(0x02441453)))); \
  d = ROTATELEFT_SIMD_9(d); \
  d = vaddq_u32(d, a); \
  \
  /* 步骤7 */ \
  c = vaddq_u32(c, vaddq_u32(G_SIMD(d, a, b), vaddq_u32(x_vec[15], vdupq_n_u32(0xd8a1e681)))); \
  c = ROTATELEFT_SIMD_14(c); \
  c = vaddq_u32(c, d); \
  \
  /* 步骤8 */ \
  b = vaddq_u32(b, vaddq_u32(G_SIMD(c, d, a), vaddq_u32(x_vec[4], vdupq_n_u32(0xe7d3fbc8)))); \
  b = ROTATELEFT_SIMD_20(b); \
  b = vaddq_u32(b, c); \
  \
  /* 步骤9 */ \
  a = vaddq_u32(a, vaddq_u32(G_SIMD(b, c, d), vaddq_u32(x_vec[9], vdupq_n_u32(0x21e1cde6)))); \
  a = ROTATELEFT_SIMD_5(a); \
  a = vaddq_u32(a, b); \
  \
  /* 步骤10 */ \
  d = vaddq_u32(d, vaddq_u32(G_SIMD(a, b, c), vaddq_u32(x_vec[14], vdupq_n_u32(0xc33707d6)))); \
  d = ROTATELEFT_SIMD_9(d); \
  d = vaddq_u32(d, a); \
  \
  /* 步骤11 */ \
  c = vaddq_u32(c, vaddq_u32(G_SIMD(d, a, b), vaddq_u32(x_vec[3], vdupq_n_u32(0xf4d50d87)))); \
  c = ROTATELEFT_SIMD_14(c); \
  c = vaddq_u32(c, d); \
  \
  /* 步骤12 */ \
  b = vaddq_u32(b, vaddq_u32(G_SIMD(c, d, a), vaddq_u32(x_vec[8], vdupq_n_u32(0x455a14ed)))); \
  b = ROTATELEFT_SIMD_20(b); \
  b = vaddq_u32(b, c); \
  \
  /* 步骤13 */ \
  a = vaddq_u32(a, vaddq_u32(G_SIMD(b, c, d), vaddq_u32(x_vec[13], vdupq_n_u32(0xa9e3e905)))); \
  a = ROTATELEFT_SIMD_5(a); \
  a = vaddq_u32(a, b); \
  \
  /* 步骤14 */ \
  d = vaddq_u32(d, vaddq_u32(G_SIMD(a, b, c), vaddq_u32(x_vec[2], vdupq_n_u32(0xfcefa3f8)))); \
  d = ROTATELEFT_SIMD_9(d); \
  d = vaddq_u32(d, a); \
  \
  /* 步骤15 */ \
  c = vaddq_u32(c, vaddq_u32(G_SIMD(d, a, b), vaddq_u32(x_vec[7], vdupq_n_u32(0x676f02d9)))); \
  c = ROTATELEFT_SIMD_14(c); \
  c = vaddq_u32(c, d); \
  \
  /* 步骤16 */ \
  b = vaddq_u32(b, vaddq_u32(G_SIMD(c, d, a), vaddq_u32(x_vec[12], vdupq_n_u32(0x8d2a4c8a)))); \
  b = ROTATELEFT_SIMD_20(b); \
  b = vaddq_u32(b, c);

#define HH_ROUND_FULLY_UNROLLED(a, b, c, d, x_vec) \
  /* 步骤1 */ \
  a = vaddq_u32(a, vaddq_u32(H_SIMD(b, c, d), vaddq_u32(x_vec[5], vdupq_n_u32(0xfffa3942)))); \
  a = ROTATELEFT_SIMD_4(a); \
  a = vaddq_u32(a, b); \
  \
  /* 步骤2 */ \
  d = vaddq_u32(d, vaddq_u32(H_SIMD(a, b, c), vaddq_u32(x_vec[8], vdupq_n_u32(0x8771f681)))); \
  d = ROTATELEFT_SIMD_11(d); \
  d = vaddq_u32(d, a); \
  \
  /* 步骤3 */ \
  c = vaddq_u32(c, vaddq_u32(H_SIMD(d, a, b), vaddq_u32(x_vec[11], vdupq_n_u32(0x6d9d6122)))); \
  c = ROTATELEFT_SIMD_16(c); \
  c = vaddq_u32(c, d); \
  \
  /* 步骤4 */ \
  b = vaddq_u32(b, vaddq_u32(H_SIMD(c, d, a), vaddq_u32(x_vec[14], vdupq_n_u32(0xfde5380c)))); \
  b = ROTATELEFT_SIMD_23(b); \
  b = vaddq_u32(b, c); \
  \
  /* 步骤5 */ \
  a = vaddq_u32(a, vaddq_u32(H_SIMD(b, c, d), vaddq_u32(x_vec[1], vdupq_n_u32(0xa4beea44)))); \
  a = ROTATELEFT_SIMD_4(a); \
  a = vaddq_u32(a, b); \
  \
  /* 步骤6 */ \
  d = vaddq_u32(d, vaddq_u32(H_SIMD(a, b, c), vaddq_u32(x_vec[4], vdupq_n_u32(0x4bdecfa9)))); \
  d = ROTATELEFT_SIMD_11(d); \
  d = vaddq_u32(d, a); \
  \
  /* 步骤7 */ \
  c = vaddq_u32(c, vaddq_u32(H_SIMD(d, a, b), vaddq_u32(x_vec[7], vdupq_n_u32(0xf6bb4b60)))); \
  c = ROTATELEFT_SIMD_16(c); \
  c = vaddq_u32(c, d); \
  \
  /* 步骤8 */ \
  b = vaddq_u32(b, vaddq_u32(H_SIMD(c, d, a), vaddq_u32(x_vec[10], vdupq_n_u32(0xbebfbc70)))); \
  b = ROTATELEFT_SIMD_23(b); \
  b = vaddq_u32(b, c); \
  \
  /* 步骤9 */ \
  a = vaddq_u32(a, vaddq_u32(H_SIMD(b, c, d), vaddq_u32(x_vec[13], vdupq_n_u32(0x289b7ec6)))); \
  a = ROTATELEFT_SIMD_4(a); \
  a = vaddq_u32(a, b); \
  \
  /* 步骤10 */ \
  d = vaddq_u32(d, vaddq_u32(H_SIMD(a, b, c), vaddq_u32(x_vec[0], vdupq_n_u32(0xeaa127fa)))); \
  d = ROTATELEFT_SIMD_11(d); \
  d = vaddq_u32(d, a); \
  \
  /* 步骤11 */ \
  c = vaddq_u32(c, vaddq_u32(H_SIMD(d, a, b), vaddq_u32(x_vec[3], vdupq_n_u32(0xd4ef3085)))); \
  c = ROTATELEFT_SIMD_16(c); \
  c = vaddq_u32(c, d); \
  \
  /* 步骤12 */ \
  b = vaddq_u32(b, vaddq_u32(H_SIMD(c, d, a), vaddq_u32(x_vec[6], vdupq_n_u32(0x04881d05)))); \
  b = ROTATELEFT_SIMD_23(b); \
  b = vaddq_u32(b, c); \
  \
  /* 步骤13 */ \
  a = vaddq_u32(a, vaddq_u32(H_SIMD(b, c, d), vaddq_u32(x_vec[9], vdupq_n_u32(0xd9d4d039)))); \
  a = ROTATELEFT_SIMD_4(a); \
  a = vaddq_u32(a, b); \
  \
  /* 步骤14 */ \
  d = vaddq_u32(d, vaddq_u32(H_SIMD(a, b, c), vaddq_u32(x_vec[12], vdupq_n_u32(0xe6db99e5)))); \
  d = ROTATELEFT_SIMD_11(d); \
  d = vaddq_u32(d, a); \
  \
  /* 步骤15 */ \
  c = vaddq_u32(c, vaddq_u32(H_SIMD(d, a, b), vaddq_u32(x_vec[15], vdupq_n_u32(0x1fa27cf8)))); \
  c = ROTATELEFT_SIMD_16(c); \
  c = vaddq_u32(c, d); \
  \
  /* 步骤16 */ \
  b = vaddq_u32(b, vaddq_u32(H_SIMD(c, d, a), vaddq_u32(x_vec[2], vdupq_n_u32(0xc4ac5665)))); \
  b = ROTATELEFT_SIMD_23(b); \
  b = vaddq_u32(b, c);

#define II_ROUND_FULLY_UNROLLED(a, b, c, d, x_vec) \
  /* 步骤1 */ \
  a = vaddq_u32(a, vaddq_u32(I_SIMD(b, c, d), vaddq_u32(x_vec[0], vdupq_n_u32(0xf4292244)))); \
  a = ROTATELEFT_SIMD_6(a); \
  a = vaddq_u32(a, b); \
  \
  /* 步骤2 */ \
  d = vaddq_u32(d, vaddq_u32(I_SIMD(a, b, c), vaddq_u32(x_vec[7], vdupq_n_u32(0x432aff97)))); \
  d = ROTATELEFT_SIMD_10(d); \
  d = vaddq_u32(d, a); \
  \
  /* 步骤3 */ \
  c = vaddq_u32(c, vaddq_u32(I_SIMD(d, a, b), vaddq_u32(x_vec[14], vdupq_n_u32(0xab9423a7)))); \
  c = ROTATELEFT_SIMD_15(c); \
  c = vaddq_u32(c, d); \
  \
  /* 步骤4 */ \
  b = vaddq_u32(b, vaddq_u32(I_SIMD(c, d, a), vaddq_u32(x_vec[5], vdupq_n_u32(0xfc93a039)))); \
  b = ROTATELEFT_SIMD_21(b); \
  b = vaddq_u32(b, c); \
  \
  /* 步骤5 */ \
  a = vaddq_u32(a, vaddq_u32(I_SIMD(b, c, d), vaddq_u32(x_vec[12], vdupq_n_u32(0x655b59c3)))); \
  a = ROTATELEFT_SIMD_6(a); \
  a = vaddq_u32(a, b); \
  \
  /* 步骤6 */ \
  d = vaddq_u32(d, vaddq_u32(I_SIMD(a, b, c), vaddq_u32(x_vec[3], vdupq_n_u32(0x8f0ccc92)))); \
  d = ROTATELEFT_SIMD_10(d); \
  d = vaddq_u32(d, a); \
  \
  /* 步骤7 */ \
  c = vaddq_u32(c, vaddq_u32(I_SIMD(d, a, b), vaddq_u32(x_vec[10], vdupq_n_u32(0xffeff47d)))); \
  c = ROTATELEFT_SIMD_15(c); \
  c = vaddq_u32(c, d); \
  \
  /* 步骤8 */ \
  b = vaddq_u32(b, vaddq_u32(I_SIMD(c, d, a), vaddq_u32(x_vec[1], vdupq_n_u32(0x85845dd1)))); \
  b = ROTATELEFT_SIMD_21(b); \
  b = vaddq_u32(b, c); \
  \
  /* 步骤9 */ \
  a = vaddq_u32(a, vaddq_u32(I_SIMD(b, c, d), vaddq_u32(x_vec[8], vdupq_n_u32(0x6fa87e4f)))); \
  a = ROTATELEFT_SIMD_6(a); \
  a = vaddq_u32(a, b); \
  \
  /* 步骤10 */ \
  d = vaddq_u32(d, vaddq_u32(I_SIMD(a, b, c), vaddq_u32(x_vec[15], vdupq_n_u32(0xfe2ce6e0)))); \
  d = ROTATELEFT_SIMD_10(d); \
  d = vaddq_u32(d, a); \
  \
  /* 步骤11 */ \
  c = vaddq_u32(c, vaddq_u32(I_SIMD(d, a, b), vaddq_u32(x_vec[6], vdupq_n_u32(0xa3014314)))); \
  c = ROTATELEFT_SIMD_15(c); \
  c = vaddq_u32(c, d); \
  \
  /* 步骤12 */ \
  b = vaddq_u32(b, vaddq_u32(I_SIMD(c, d, a), vaddq_u32(x_vec[13], vdupq_n_u32(0x4e0811a1)))); \
  b = ROTATELEFT_SIMD_21(b); \
  b = vaddq_u32(b, c); \
  \
  /* 步骤13 */ \
  a = vaddq_u32(a, vaddq_u32(I_SIMD(b, c, d), vaddq_u32(x_vec[4], vdupq_n_u32(0xf7537e82)))); \
  a = ROTATELEFT_SIMD_6(a); \
  a = vaddq_u32(a, b); \
  \
  /* 步骤14 */ \
  d = vaddq_u32(d, vaddq_u32(I_SIMD(a, b, c), vaddq_u32(x_vec[11], vdupq_n_u32(0xbd3af235)))); \
  d = ROTATELEFT_SIMD_10(d); \
  d = vaddq_u32(d, a); \
  \
  /* 步骤15 */ \
  c = vaddq_u32(c, vaddq_u32(I_SIMD(d, a, b), vaddq_u32(x_vec[2], vdupq_n_u32(0x2ad7d2bb)))); \
  c = ROTATELEFT_SIMD_15(c); \
  c = vaddq_u32(c, d); \
  \
  /* 步骤16 */ \
  b = vaddq_u32(b, vaddq_u32(I_SIMD(c, d, a), vaddq_u32(x_vec[9], vdupq_n_u32(0xeb86d391)))); \
  b = ROTATELEFT_SIMD_21(b); \
  b = vaddq_u32(b, c);

void MD5Hash_SIMD(vector<string>& inputs, const int* lengths, const int num_inputs, bit32 states[][4]);
void MD5Hash_SIMD4(const string inputs[4],bit32 states[4][4]);
void MD5Hash_SIMD8(const string inputs[8],bit32 states[8][4]);
void MD5Hash_SIMD12(const string inputs[12],bit32 states[12][4]);
void MD5Hash_SIMD16(const string input[16], bit32 states[16][4]);
