#include "md5.h"
#include <iomanip>
#include <assert.h>
#include <chrono>
#include <algorithm>
using namespace std;
using namespace chrono;

/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组
 * @param input 输入
 * @param[out] n_byte 用于给调用者传递额外的返回值，即最终Byte数组的长度
 * @return Byte消息数组
 */
Byte *StringProcess(string input, int *n_byte)
{
	// 将输入的字符串转换为Byte为单位的数组
	Byte *blocks = (Byte *)input.c_str();
	int length = input.length();

	// 计算原始消息长度（以比特为单位）
	int bitLength = length * 8;
	int paddingBytes = ((448 - bitLength % 512 + 511) % 512 + 1) / 8;
	// paddingBits: 原始消息需要的padding长度（以bit为单位）
	// 对于给定的消息，将其补齐至length%512==448为止
	// 需要注意的是，即便给定的消息满足length%512==448，也需要再pad 512bits
	//  int paddingBits = bitLength % 512;
	//  if (paddingBits > 448)
	//  {
	//  	paddingBits += 512 - (paddingBits - 448);
	//  }
	//  else if (paddingBits < 448)
	//  {
	//  	paddingBits = 448 - paddingBits;
	//  }
	//  else if (paddingBits == 448)
	//  {
	//  	paddingBits = 512;
	//  }

	//  // 原始消息需要的padding长度（以Byte为单位）
	//  int paddingBytes = paddingBits / 8;
	// 创建最终的字节数组
	// length + paddingBytes + 8:
	// 1. length为原始消息的长度（bits）
	// 2. paddingBytes为原始消息需要的padding长度（Bytes）
	// 3. 在pad到length%512==448之后，需要额外附加64bits的原始消息长度，即8个bytes
	int paddedLength = length + paddingBytes + 8;
	Byte *paddedMessage = new Byte[paddedLength];

	// 复制原始消息
	memcpy(paddedMessage, blocks, length);

	// 添加填充字节。填充时，第一位为1，后面的所有位均为0。
	// 所以第一个byte是0x80
	paddedMessage[length] = 0x80;							 // 添加一个0x80字节
	memset(paddedMessage + length + 1, 0, paddingBytes - 1); // 填充0字节

	// 添加消息长度（64比特，小端格式）
	for (int i = 0; i < 8; ++i)
	{
		// 特别注意此处应当将bitLength转换为uint64_t
		// 这里的length是原始消息的长度
		paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
	}

	// 验证长度是否满足要求。此时长度应当是512bit的倍数
	int residual = 8 * paddedLength % 512;
	// assert(residual == 0);

	// 在填充+添加长度之后，消息被分为n_blocks个512bit的部分
	*n_byte = paddedLength;
	return paddedMessage;
}

/**
 * MD5Hash: 将单个输入字符串转换成MD5
 * @param input 输入
 * @param[out] state 用于给调用者传递额外的返回值，即最终的缓冲区，也就是MD5的结果
 * @return Byte消息数组
 */
void MD5Hash(string input, bit32 *state)
{

	Byte *paddedMessage;
	int *messageLength = new int[1];
	for (int i = 0; i < 1; i += 1)
	{
		paddedMessage = StringProcess(input, &messageLength[i]);
		// cout<<messageLength[i]<<endl;
		assert(messageLength[i] == messageLength[0]);
	}
	int n_blocks = messageLength[0] / 64;

	// bit32* state= new bit32[4];
	state[0] = 0x67452301;
	state[1] = 0xefcdab89;
	state[2] = 0x98badcfe;
	state[3] = 0x10325476;

	// 逐block地更新state
	for (int i = 0; i < n_blocks; i += 1)
	{
		bit32 x[16];

		// 下面的处理，在理解上较为复杂
		for (int i1 = 0; i1 < 16; ++i1)
		{
			x[i1] = (paddedMessage[4 * i1 + i * 64]) |
					(paddedMessage[4 * i1 + 1 + i * 64] << 8) |
					(paddedMessage[4 * i1 + 2 + i * 64] << 16) |
					(paddedMessage[4 * i1 + 3 + i * 64] << 24);
		}

		bit32 a = state[0], b = state[1], c = state[2], d = state[3];

		auto start = system_clock::now();
		/* Round 1 */
		FF(a, b, c, d, x[0], s11, 0xd76aa478);
		FF(d, a, b, c, x[1], s12, 0xe8c7b756);
		FF(c, d, a, b, x[2], s13, 0x242070db);
		FF(b, c, d, a, x[3], s14, 0xc1bdceee);
		FF(a, b, c, d, x[4], s11, 0xf57c0faf);
		FF(d, a, b, c, x[5], s12, 0x4787c62a);
		FF(c, d, a, b, x[6], s13, 0xa8304613);
		FF(b, c, d, a, x[7], s14, 0xfd469501);
		FF(a, b, c, d, x[8], s11, 0x698098d8);
		FF(d, a, b, c, x[9], s12, 0x8b44f7af);
		FF(c, d, a, b, x[10], s13, 0xffff5bb1);
		FF(b, c, d, a, x[11], s14, 0x895cd7be);
		FF(a, b, c, d, x[12], s11, 0x6b901122);
		FF(d, a, b, c, x[13], s12, 0xfd987193);
		FF(c, d, a, b, x[14], s13, 0xa679438e);
		FF(b, c, d, a, x[15], s14, 0x49b40821);

		/* Round 2 */
		GG(a, b, c, d, x[1], s21, 0xf61e2562);
		GG(d, a, b, c, x[6], s22, 0xc040b340);
		GG(c, d, a, b, x[11], s23, 0x265e5a51);
		GG(b, c, d, a, x[0], s24, 0xe9b6c7aa);
		GG(a, b, c, d, x[5], s21, 0xd62f105d);
		GG(d, a, b, c, x[10], s22, 0x02441453);
		GG(c, d, a, b, x[15], s23, 0xd8a1e681);
		GG(b, c, d, a, x[4], s24, 0xe7d3fbc8);
		GG(a, b, c, d, x[9], s21, 0x21e1cde6);
		GG(d, a, b, c, x[14], s22, 0xc33707d6);
		GG(c, d, a, b, x[3], s23, 0xf4d50d87);
		GG(b, c, d, a, x[8], s24, 0x455a14ed);
		GG(a, b, c, d, x[13], s21, 0xa9e3e905);
		GG(d, a, b, c, x[2], s22, 0xfcefa3f8);
		GG(c, d, a, b, x[7], s23, 0x676f02d9);
		GG(b, c, d, a, x[12], s24, 0x8d2a4c8a);

		/* Round 3 */
		HH(a, b, c, d, x[5], s31, 0xfffa3942);
		HH(d, a, b, c, x[8], s32, 0x8771f681);
		HH(c, d, a, b, x[11], s33, 0x6d9d6122);
		HH(b, c, d, a, x[14], s34, 0xfde5380c);
		HH(a, b, c, d, x[1], s31, 0xa4beea44);
		HH(d, a, b, c, x[4], s32, 0x4bdecfa9);
		HH(c, d, a, b, x[7], s33, 0xf6bb4b60);
		HH(b, c, d, a, x[10], s34, 0xbebfbc70);
		HH(a, b, c, d, x[13], s31, 0x289b7ec6);
		HH(d, a, b, c, x[0], s32, 0xeaa127fa);
		HH(c, d, a, b, x[3], s33, 0xd4ef3085);
		HH(b, c, d, a, x[6], s34, 0x04881d05);
		HH(a, b, c, d, x[9], s31, 0xd9d4d039);
		HH(d, a, b, c, x[12], s32, 0xe6db99e5);
		HH(c, d, a, b, x[15], s33, 0x1fa27cf8);
		HH(b, c, d, a, x[2], s34, 0xc4ac5665);

		/* Round 4 */
		II(a, b, c, d, x[0], s41, 0xf4292244);
		II(d, a, b, c, x[7], s42, 0x432aff97);
		II(c, d, a, b, x[14], s43, 0xab9423a7);
		II(b, c, d, a, x[5], s44, 0xfc93a039);
		II(a, b, c, d, x[12], s41, 0x655b59c3);
		II(d, a, b, c, x[3], s42, 0x8f0ccc92);
		II(c, d, a, b, x[10], s43, 0xffeff47d);
		II(b, c, d, a, x[1], s44, 0x85845dd1);
		II(a, b, c, d, x[8], s41, 0x6fa87e4f);
		II(d, a, b, c, x[15], s42, 0xfe2ce6e0);
		II(c, d, a, b, x[6], s43, 0xa3014314);
		II(b, c, d, a, x[13], s44, 0x4e0811a1);
		II(a, b, c, d, x[4], s41, 0xf7537e82);
		II(d, a, b, c, x[11], s42, 0xbd3af235);
		II(c, d, a, b, x[2], s43, 0x2ad7d2bb);
		II(b, c, d, a, x[9], s44, 0xeb86d391);

		state[0] += a;
		state[1] += b;
		state[2] += c;
		state[3] += d;
	}

	// 下面的处理，在理解上较为复杂
	for (int i = 0; i < 4; i++)
	{
		uint32_t value = state[i];
		state[i] = ((value & 0xff) << 24) |		 // 将最低字节移到最高位
				   ((value & 0xff00) << 8) |	 // 将次低字节左移
				   ((value & 0xff0000) >> 8) |	 // 将次高字节右移
				   ((value & 0xff000000) >> 24); // 将最高字节移到最低位
	}

	// 输出最终的hash结果
	// for (int i1 = 0; i1 < 4; i1 += 1)
	// {
	// 	cout << std::setw(8) << std::setfill('0') << hex << state[i1];
	// }
	// cout << endl;

	// 释放动态分配的内存
	// 实现SIMD并行算法的时候，也请记得及时回收内存！
	delete[] paddedMessage;
	delete[] messageLength;
}

void MD5Hash_SIMD(vector<string> &inputs, const int *lengths, const int num_inputs, bit32 states[][4])
{
	// NEON一次处理4个数据,所以向上对齐到4的倍数
	const int n_inputs = ((num_inputs + 3) / 4) * 4;

	// 初始化数据结构
	vector<unique_ptr<Byte[]>> paddedMessages(num_inputs);
	vector<int> paddedLengths(num_inputs);
	vector<int> n_blocks(num_inputs);
	vector<bit32> x(n_inputs * 16, 0);

// 预处理输入
#pragma omp parallel for
	for (int j = 0; j < num_inputs; ++j)
	{
		paddedMessages[j].reset(StringProcess(inputs[j], &paddedLengths[j]));
		n_blocks[j] = paddedLengths[j] / 64;
	}

	// 对齐内存分配
	bit32 *state0 = (bit32 *)aligned_alloc(16, n_inputs * sizeof(bit32));
	bit32 *state1 = (bit32 *)aligned_alloc(16, n_inputs * sizeof(bit32));
	bit32 *state2 = (bit32 *)aligned_alloc(16, n_inputs * sizeof(bit32));
	bit32 *state3 = (bit32 *)aligned_alloc(16, n_inputs * sizeof(bit32));

	state0[0] = 0x67452301;
	state1[0] = 0xefcdab89;
	state2[0] = 0x98badcfe;
	state3[0] = 0x10325476;

	// 使用内存复制扩展法快速填充
	int filled = 1;
	while (filled * 2 <= n_inputs)
	{
		memcpy(state0 + filled, state0, filled * sizeof(bit32));
		memcpy(state1 + filled, state1, filled * sizeof(bit32));
		memcpy(state2 + filled, state2, filled * sizeof(bit32));
		memcpy(state3 + filled, state3, filled * sizeof(bit32));
		filled *= 2;
	}

	// 处理剩余元素
	if (filled < n_inputs)
	{
		memcpy(state0 + filled, state0, (n_inputs - filled) * sizeof(bit32));
		memcpy(state1 + filled, state1, (n_inputs - filled) * sizeof(bit32));
		memcpy(state2 + filled, state2, (n_inputs - filled) * sizeof(bit32));
		memcpy(state3 + filled, state3, (n_inputs - filled) * sizeof(bit32));
	}

	// 初始化state
	//   for (int i = 0; i < n_inputs; i++)
	//   {
	//   	state0[i] = 0x67452301;
	//   	state1[i] = 0xefcdab89;
	//   	state2[i] = 0x98badcfe;
	//   	state3[i] = 0x10325476;
	//   }

	int max_blocks = *max_element(n_blocks.begin(), n_blocks.end());

	for (int i = 0; i < max_blocks; ++i)
	{
		for (int j = 0; j < num_inputs; ++j)
		{
			if (i >= n_blocks[j])
				continue;
			for (int k = 0; k < 16; ++k)
			{
				size_t msg_offset = i * 64 + k * 4;
				size_t x_offset = k * n_inputs + j;
				/// size_t x_offset =j*16+k;
				x[x_offset] = (paddedMessages[j][msg_offset] |
							   (paddedMessages[j][msg_offset + 1] << 8) |
							   (paddedMessages[j][msg_offset + 2] << 16) |
							   (paddedMessages[j][msg_offset + 3] << 24));
			}
		}

		// 每次处理4个数据
		for (int j = 0; j < n_inputs; j += 4)
		{
			uint32x4_t x_vec[16];
			////加载x数据
			for (int k = 0; k < 16; ++k)
			{
				x_vec[k] = vld1q_u32(&x[k * n_inputs + j]);
			}
			//   for (int k = 0; k < 16; ++k) {
			//   	uint32x4_t temp = {
			//   		x[j * 16 + k],
			//   		x[(j+1) * 16 + k],
			//   		x[(j+2) * 16 + k],
			//   		x[(j+3) * 16 + k]
			//   	};
			//   	x_vec[k] = temp;
			//   }

			// 加载state
			uint32x4_t a = vld1q_u32(state0 + j);
			uint32x4_t b = vld1q_u32(state1 + j);
			uint32x4_t c = vld1q_u32(state2 + j);
			uint32x4_t d = vld1q_u32(state3 + j);

			uint32x4_t aa = a, bb = b, cc = c, dd = d;

			//  FF_round_fully_unrolled(a, b, c, d, x_vec);
			//  GG_round_fully_unrolled(a, b, c, d, x_vec);
			//  HH_round_fully_unrolled(a, b, c, d, x_vec);
			//  II_round_fully_unrolled(a, b, c, d, x_vec);

			FF_SIMD(a, b, c, d, x_vec[0], s11, 0xd76aa478);
			FF_SIMD(d, a, b, c, x_vec[1], s12, 0xe8c7b756);
			FF_SIMD(c, d, a, b, x_vec[2], s13, 0x242070db);
			FF_SIMD(b, c, d, a, x_vec[3], s14, 0xc1bdceee);
			FF_SIMD(a, b, c, d, x_vec[4], s11, 0xf57c0faf);
			FF_SIMD(d, a, b, c, x_vec[5], s12, 0x4787c62a);
			FF_SIMD(c, d, a, b, x_vec[6], s13, 0xa8304613);
			FF_SIMD(b, c, d, a, x_vec[7], s14, 0xfd469501);
			FF_SIMD(a, b, c, d, x_vec[8], s11, 0x698098d8);
			FF_SIMD(d, a, b, c, x_vec[9], s12, 0x8b44f7af);
			FF_SIMD(c, d, a, b, x_vec[10], s13, 0xffff5bb1);
			FF_SIMD(b, c, d, a, x_vec[11], s14, 0x895cd7be);
			FF_SIMD(a, b, c, d, x_vec[12], s11, 0x6b901122);
			FF_SIMD(d, a, b, c, x_vec[13], s12, 0xfd987193);
			FF_SIMD(c, d, a, b, x_vec[14], s13, 0xa679438e);
			FF_SIMD(b, c, d, a, x_vec[15], s14, 0x49b40821);

			GG_SIMD(a, b, c, d, x_vec[1], s21, 0xf61e2562);
			GG_SIMD(d, a, b, c, x_vec[6], s22, 0xc040b340);
			GG_SIMD(c, d, a, b, x_vec[11], s23, 0x265e5a51);
			GG_SIMD(b, c, d, a, x_vec[0], s24, 0xe9b6c7aa);
			GG_SIMD(a, b, c, d, x_vec[5], s21, 0xd62f105d);
			GG_SIMD(d, a, b, c, x_vec[10], s22, 0x02441453);
			GG_SIMD(c, d, a, b, x_vec[15], s23, 0xd8a1e681);
			GG_SIMD(b, c, d, a, x_vec[4], s24, 0xe7d3fbc8);
			GG_SIMD(a, b, c, d, x_vec[9], s21, 0x21e1cde6);
			GG_SIMD(d, a, b, c, x_vec[14], s22, 0xc33707d6);
			GG_SIMD(c, d, a, b, x_vec[3], s23, 0xf4d50d87);
			GG_SIMD(b, c, d, a, x_vec[8], s24, 0x455a14ed);
			GG_SIMD(a, b, c, d, x_vec[13], s21, 0xa9e3e905);
			GG_SIMD(d, a, b, c, x_vec[2], s22, 0xfcefa3f8);
			GG_SIMD(c, d, a, b, x_vec[7], s23, 0x676f02d9);
			GG_SIMD(b, c, d, a, x_vec[12], s24, 0x8d2a4c8a);

			HH_SIMD(a, b, c, d, x_vec[5], s31, 0xfffa3942);
			HH_SIMD(d, a, b, c, x_vec[8], s32, 0x8771f681);
			HH_SIMD(c, d, a, b, x_vec[11], s33, 0x6d9d6122);
			HH_SIMD(b, c, d, a, x_vec[14], s34, 0xfde5380c);
			HH_SIMD(a, b, c, d, x_vec[1], s31, 0xa4beea44);
			HH_SIMD(d, a, b, c, x_vec[4], s32, 0x4bdecfa9);
			HH_SIMD(c, d, a, b, x_vec[7], s33, 0xf6bb4b60);
			HH_SIMD(b, c, d, a, x_vec[10], s34, 0xbebfbc70);
			HH_SIMD(a, b, c, d, x_vec[13], s31, 0x289b7ec6);
			HH_SIMD(d, a, b, c, x_vec[0], s32, 0xeaa127fa);
			HH_SIMD(c, d, a, b, x_vec[3], s33, 0xd4ef3085);
			HH_SIMD(b, c, d, a, x_vec[6], s34, 0x04881d05);
			HH_SIMD(a, b, c, d, x_vec[9], s31, 0xd9d4d039);
			HH_SIMD(d, a, b, c, x_vec[12], s32, 0xe6db99e5);
			HH_SIMD(c, d, a, b, x_vec[15], s33, 0x1fa27cf8);
			HH_SIMD(b, c, d, a, x_vec[2], s34, 0xc4ac5665);

			II_SIMD(a, b, c, d, x_vec[0], s41, 0xf4292244);
			II_SIMD(d, a, b, c, x_vec[7], s42, 0x432aff97);
			II_SIMD(c, d, a, b, x_vec[14], s43, 0xab9423a7);
			II_SIMD(b, c, d, a, x_vec[5], s44, 0xfc93a039);
			II_SIMD(a, b, c, d, x_vec[12], s41, 0x655b59c3);
			II_SIMD(d, a, b, c, x_vec[3], s42, 0x8f0ccc92);
			II_SIMD(c, d, a, b, x_vec[10], s43, 0xffeff47d);
			II_SIMD(b, c, d, a, x_vec[1], s44, 0x85845dd1);
			II_SIMD(a, b, c, d, x_vec[8], s41, 0x6fa87e4f);
			II_SIMD(d, a, b, c, x_vec[15], s42, 0xfe2ce6e0);
			II_SIMD(c, d, a, b, x_vec[6], s43, 0xa3014314);
			II_SIMD(b, c, d, a, x_vec[13], s44, 0x4e0811a1);
			II_SIMD(a, b, c, d, x_vec[4], s41, 0xf7537e82);
			II_SIMD(d, a, b, c, x_vec[11], s42, 0xbd3af235);
			II_SIMD(c, d, a, b, x_vec[2], s43, 0x2ad7d2bb);
			II_SIMD(b, c, d, a, x_vec[9], s44, 0xeb86d391);

			// 更新state
			a = vaddq_u32(a, aa);
			b = vaddq_u32(b, bb);
			c = vaddq_u32(c, cc);
			d = vaddq_u32(d, dd);

			// 存储结果
			vst1q_u32(state0 + j, a);
			vst1q_u32(state1 + j, b);
			vst1q_u32(state2 + j, c);
			vst1q_u32(state3 + j, d);
		}
	}

	// 字节序转换和结果输出
	for (int j = 0; j < num_inputs; ++j)
	{
		bit32 state[4] = {state0[j], state1[j], state2[j], state3[j]};
		for (int k = 0; k < 4; ++k)
		{
			state[k] = ((state[k] & 0xFF) << 24) |
					   ((state[k] & 0xFF00) << 8) |
					   ((state[k] & 0xFF0000) >> 8) |
					   ((state[k] & 0xFF000000) >> 24);
		}
		memcpy(states[j], state, 4 * sizeof(bit32));
	}

	// 释放内存
	free(state0);
	free(state1);
	free(state2);
	free(state3);
}
void MD5Hash_SIMD4(const string input[4], bit32 states[4][4])
{
	Byte *paddedMessages[4];
	int paddedLengths[4];
	int n_blocks[4];

	for (int i = 0; i < 4; i++)
	{
		paddedMessages[i] = StringProcess(input[i], &paddedLengths[i]);
		n_blocks[i] = paddedLengths[i] / 64;
		states[i][0] = 0x67452301;
		states[i][1] = 0xefcdab89;
		states[i][2] = 0x98badcfe;
		states[i][3] = 0x10325476;
	}
	int max_blocks = *max_element(n_blocks, n_blocks + 4);
	uint32x4_t x_vec[16];
	bit32 x[4];
	uint32x4_t a = vdupq_n_u32(states[0][0]);
	uint32x4_t b = vdupq_n_u32(states[0][1]);
	uint32x4_t c = vdupq_n_u32(states[0][2]);
	uint32x4_t d = vdupq_n_u32(states[0][3]);

	uint32x4_t aa = a, bb = b, cc = c, dd = d;
	for (int i = 0; i < max_blocks; ++i)
	{
		for (int j = 0; j < 16; j++)
		{
			int msg_offset = i * 64 + j * 4;
			x[0] = (paddedMessages[0][msg_offset]) |
				   (paddedMessages[0][msg_offset + 1] << 8) |
				   (paddedMessages[0][msg_offset + 2] << 16) |
				   (paddedMessages[0][msg_offset + 3] << 24);
			x[1] = (paddedMessages[1][msg_offset]) |
				   (paddedMessages[1][msg_offset + 1] << 8) |
				   (paddedMessages[1][msg_offset + 2] << 16) |
				   (paddedMessages[1][msg_offset + 3] << 24);
			x[2] = (paddedMessages[2][msg_offset]) |
				   (paddedMessages[2][msg_offset + 1] << 8) |
				   (paddedMessages[2][msg_offset + 2] << 16) |
				   (paddedMessages[2][msg_offset + 3] << 24);
			x[3] = (paddedMessages[3][msg_offset]) |
				   (paddedMessages[3][msg_offset + 1] << 8) |
				   (paddedMessages[3][msg_offset + 2] << 16) |
				   (paddedMessages[3][msg_offset + 3] << 24);

			//???
			x_vec[j] = vld1q_u32(x);
		}
		FF_SIMD(a, b, c, d, x_vec[0], s11, 0xd76aa478);
		FF_SIMD(d, a, b, c, x_vec[1], s12, 0xe8c7b756);
		FF_SIMD(c, d, a, b, x_vec[2], s13, 0x242070db);
		FF_SIMD(b, c, d, a, x_vec[3], s14, 0xc1bdceee);
		FF_SIMD(a, b, c, d, x_vec[4], s11, 0xf57c0faf);
		FF_SIMD(d, a, b, c, x_vec[5], s12, 0x4787c62a);
		FF_SIMD(c, d, a, b, x_vec[6], s13, 0xa8304613);
		FF_SIMD(b, c, d, a, x_vec[7], s14, 0xfd469501);
		FF_SIMD(a, b, c, d, x_vec[8], s11, 0x698098d8);
		FF_SIMD(d, a, b, c, x_vec[9], s12, 0x8b44f7af);
		FF_SIMD(c, d, a, b, x_vec[10], s13, 0xffff5bb1);
		FF_SIMD(b, c, d, a, x_vec[11], s14, 0x895cd7be);
		FF_SIMD(a, b, c, d, x_vec[12], s11, 0x6b901122);
		FF_SIMD(d, a, b, c, x_vec[13], s12, 0xfd987193);
		FF_SIMD(c, d, a, b, x_vec[14], s13, 0xa679438e);
		FF_SIMD(b, c, d, a, x_vec[15], s14, 0x49b40821);
		GG_SIMD(a, b, c, d, x_vec[1], s21, 0xf61e2562);
		GG_SIMD(d, a, b, c, x_vec[6], s22, 0xc040b340);
		GG_SIMD(c, d, a, b, x_vec[11], s23, 0x265e5a51);
		GG_SIMD(b, c, d, a, x_vec[0], s24, 0xe9b6c7aa);
		GG_SIMD(a, b, c, d, x_vec[5], s21, 0xd62f105d);
		GG_SIMD(d, a, b, c, x_vec[10], s22, 0x02441453);
		GG_SIMD(c, d, a, b, x_vec[15], s23, 0xd8a1e681);
		GG_SIMD(b, c, d, a, x_vec[4], s24, 0xe7d3fbc8);
		GG_SIMD(a, b, c, d, x_vec[9], s21, 0x21e1cde6);
		GG_SIMD(d, a, b, c, x_vec[14], s22, 0xc33707d6);
		GG_SIMD(c, d, a, b, x_vec[3], s23, 0xf4d50d87);
		GG_SIMD(b, c, d, a, x_vec[8], s24, 0x455a14ed);
		GG_SIMD(a, b, c, d, x_vec[13], s21, 0xa9e3e905);
		GG_SIMD(d, a, b, c, x_vec[2], s22, 0xfcefa3f8);
		GG_SIMD(c, d, a, b, x_vec[7], s23, 0x676f02d9);
		GG_SIMD(b, c, d, a, x_vec[12], s24, 0x8d2a4c8a);
		HH_SIMD(a, b, c, d, x_vec[5], s31, 0xfffa3942);
		HH_SIMD(d, a, b, c, x_vec[8], s32, 0x8771f681);
		HH_SIMD(c, d, a, b, x_vec[11], s33, 0x6d9d6122);
		HH_SIMD(b, c, d, a, x_vec[14], s34, 0xfde5380c);
		HH_SIMD(a, b, c, d, x_vec[1], s31, 0xa4beea44);
		HH_SIMD(d, a, b, c, x_vec[4], s32, 0x4bdecfa9);
		HH_SIMD(c, d, a, b, x_vec[7], s33, 0xf6bb4b60);
		HH_SIMD(b, c, d, a, x_vec[10], s34, 0xbebfbc70);
		HH_SIMD(a, b, c, d, x_vec[13], s31, 0x289b7ec6);
		HH_SIMD(d, a, b, c, x_vec[0], s32, 0xeaa127fa);
		HH_SIMD(c, d, a, b, x_vec[3], s33, 0xd4ef3085);
		HH_SIMD(b, c, d, a, x_vec[6], s34, 0x04881d05);
		HH_SIMD(a, b, c, d, x_vec[9], s31, 0xd9d4d039);
		HH_SIMD(d, a, b, c, x_vec[12], s32, 0xe6db99e5);
		HH_SIMD(c, d, a, b, x_vec[15], s33, 0x1fa27cf8);
		HH_SIMD(b, c, d, a, x_vec[2], s34, 0xc4ac5665);
		II_SIMD(a, b, c, d, x_vec[0], s41, 0xf4292244);
		II_SIMD(d, a, b, c, x_vec[7], s42, 0x432aff97);
		II_SIMD(c, d, a, b, x_vec[14], s43, 0xab9423a7);
		II_SIMD(b, c, d, a, x_vec[5], s44, 0xfc93a039);
		II_SIMD(a, b, c, d, x_vec[12], s41, 0x655b59c3);
		II_SIMD(d, a, b, c, x_vec[3], s42, 0x8f0ccc92);
		II_SIMD(c, d, a, b, x_vec[10], s43, 0xffeff47d);
		II_SIMD(b, c, d, a, x_vec[1], s44, 0x85845dd1);
		II_SIMD(a, b, c, d, x_vec[8], s41, 0x6fa87e4f);
		II_SIMD(d, a, b, c, x_vec[15], s42, 0xfe2ce6e0);
		II_SIMD(c, d, a, b, x_vec[6], s43, 0xa3014314);
		II_SIMD(b, c, d, a, x_vec[13], s44, 0x4e0811a1);
		II_SIMD(a, b, c, d, x_vec[4], s41, 0xf7537e82);
		II_SIMD(d, a, b, c, x_vec[11], s42, 0xbd3af235);
		II_SIMD(c, d, a, b, x_vec[2], s43, 0x2ad7d2bb);
		II_SIMD(b, c, d, a, x_vec[9], s44, 0xeb86d391);

		a = vaddq_u32(a, aa);
		b = vaddq_u32(b, bb);
		c = vaddq_u32(c, cc);
		d = vaddq_u32(d, dd);
		//  a=aa;
		//  b=bb;
		//  c=cc;
		//  d=dd;
	}
	uint32_t result_a[4], result_b[4], result_c[4], result_d[4];
	vst1q_u32(result_a, a);
	vst1q_u32(result_b, b);
	vst1q_u32(result_c, c);
	vst1q_u32(result_d, d);

	for (int i = 0; i < 4; i++)
	{
		states[i][0] = ((result_a[i] & 0xFF) << 24) |
					   ((result_a[i] & 0xFF00) << 8) |
					   ((result_a[i] & 0xFF0000) >> 8) |
					   ((result_a[i] & 0xFF000000) >> 24);

		states[i][1] = ((result_b[i] & 0xFF) << 24) |
					   ((result_b[i] & 0xFF00) << 8) |
					   ((result_b[i] & 0xFF0000) >> 8) |
					   ((result_b[i] & 0xFF000000) >> 24);

		states[i][2] = ((result_c[i] & 0xFF) << 24) |
					   ((result_c[i] & 0xFF00) << 8) |
					   ((result_c[i] & 0xFF0000) >> 8) |
					   ((result_c[i] & 0xFF000000) >> 24);

		states[i][3] = ((result_d[i] & 0xFF) << 24) |
					   ((result_d[i] & 0xFF00) << 8) |
					   ((result_d[i] & 0xFF0000) >> 8) |
					   ((result_d[i] & 0xFF000000) >> 24);
	}
}

// 8路并行
void MD5Hash_SIMD8(const string input[8], bit32 states[8][4])
{
	// 为8个输入准备缓冲区
	Byte *paddedMessages[8];
	int paddedLengths[8];
	int n_blocks[8];

	// 处理每个字符串，初始化状态
	for (int i = 0; i < 8; i++)
	{
		paddedMessages[i] = StringProcess(input[i], &paddedLengths[i]);
		n_blocks[i] = paddedLengths[i] / 64;
		states[i][0] = 0x67452301;
		states[i][1] = 0xefcdab89;
		states[i][2] = 0x98badcfe;
		states[i][3] = 0x10325476;
	}

	// 获取最大块数
	int max_blocks = *max_element(n_blocks, n_blocks + 8);

	// 创建两组4个字符串的状态变量
	uint32x4_t a1 = vdupq_n_u32(0x67452301);
	uint32x4_t b1 = vdupq_n_u32(0xefcdab89);
	uint32x4_t c1 = vdupq_n_u32(0x98badcfe);
	uint32x4_t d1 = vdupq_n_u32(0x10325476);

	uint32x4_t a2 = vdupq_n_u32(0x67452301);
	uint32x4_t b2 = vdupq_n_u32(0xefcdab89);
	uint32x4_t c2 = vdupq_n_u32(0x98badcfe);
	uint32x4_t d2 = vdupq_n_u32(0x10325476);

	uint32x4_t aa1 = a1, bb1 = b1, cc1 = c1, dd1 = d1;
	uint32x4_t aa2 = a2, bb2 = b2, cc2 = c2, dd2 = d2;

	// 临时缓冲区
	uint32x4_t x_vec1[16], x_vec2[16];
	bit32 x1[4], x2[4];

	// 处理每个块
	for (int i = 0; i < max_blocks; ++i)
	{
		// 重置状态为保存的初始值
		a1 = aa1;
		b1 = bb1;
		c1 = cc1;
		d1 = dd1;
		a2 = aa2;
		b2 = bb2;
		c2 = cc2;
		d2 = dd2;

		// 为两批4个字符串分别加载消息块
		for (int j = 0; j < 16; j++)
		{
			int msg_offset = i * 64 + j * 4;

			// 处理第一组4个字符串
			for (int k = 0; k < 4; k++)
			{
				if (i < n_blocks[k])
				{
					x1[k] = (paddedMessages[k][msg_offset]) |
							(paddedMessages[k][msg_offset + 1] << 8) |
							(paddedMessages[k][msg_offset + 2] << 16) |
							(paddedMessages[k][msg_offset + 3] << 24);
				}
				else
				{
					// 如果这个输入已处理完所有块，使用最后一个有效块或零
					x1[k] = 0;
				}
			}

			// 处理第二组4个字符串
			for (int k = 0; k < 4; k++)
			{
				if (i < n_blocks[k + 4])
				{
					x2[k] = (paddedMessages[k + 4][msg_offset]) |
							(paddedMessages[k + 4][msg_offset + 1] << 8) |
							(paddedMessages[k + 4][msg_offset + 2] << 16) |
							(paddedMessages[k + 4][msg_offset + 3] << 24);
				}
				else
				{
					// 如果这个输入已处理完所有块，使用最后一个有效块或零
					x2[k] = 0;
				}
			}

			x_vec1[j] = vld1q_u32(x1);
			x_vec2[j] = vld1q_u32(x2);
		}

		// 现在在数据准备好后，我们将两组交织在一起执行MD5的各个步骤
		// 这实现真正的8路并行，通过同时处理两组数据

		// 第一轮 (Round 1) - 交织执行
		FF_SIMD(a1, b1, c1, d1, x_vec1[0], s11, 0xd76aa478);
		FF_SIMD(a2, b2, c2, d2, x_vec2[0], s11, 0xd76aa478);

		FF_SIMD(d1, a1, b1, c1, x_vec1[1], s12, 0xe8c7b756);
		FF_SIMD(d2, a2, b2, c2, x_vec2[1], s12, 0xe8c7b756);

		FF_SIMD(c1, d1, a1, b1, x_vec1[2], s13, 0x242070db);
		FF_SIMD(c2, d2, a2, b2, x_vec2[2], s13, 0x242070db);

		FF_SIMD(b1, c1, d1, a1, x_vec1[3], s14, 0xc1bdceee);
		FF_SIMD(b2, c2, d2, a2, x_vec2[3], s14, 0xc1bdceee);

		FF_SIMD(a1, b1, c1, d1, x_vec1[4], s11, 0xf57c0faf);
		FF_SIMD(a2, b2, c2, d2, x_vec2[4], s11, 0xf57c0faf);

		FF_SIMD(d1, a1, b1, c1, x_vec1[5], s12, 0x4787c62a);
		FF_SIMD(d2, a2, b2, c2, x_vec2[5], s12, 0x4787c62a);

		FF_SIMD(c1, d1, a1, b1, x_vec1[6], s13, 0xa8304613);
		FF_SIMD(c2, d2, a2, b2, x_vec2[6], s13, 0xa8304613);

		FF_SIMD(b1, c1, d1, a1, x_vec1[7], s14, 0xfd469501);
		FF_SIMD(b2, c2, d2, a2, x_vec2[7], s14, 0xfd469501);

		FF_SIMD(a1, b1, c1, d1, x_vec1[8], s11, 0x698098d8);
		FF_SIMD(a2, b2, c2, d2, x_vec2[8], s11, 0x698098d8);

		FF_SIMD(d1, a1, b1, c1, x_vec1[9], s12, 0x8b44f7af);
		FF_SIMD(d2, a2, b2, c2, x_vec2[9], s12, 0x8b44f7af);

		FF_SIMD(c1, d1, a1, b1, x_vec1[10], s13, 0xffff5bb1);
		FF_SIMD(c2, d2, a2, b2, x_vec2[10], s13, 0xffff5bb1);

		FF_SIMD(b1, c1, d1, a1, x_vec1[11], s14, 0x895cd7be);
		FF_SIMD(b2, c2, d2, a2, x_vec2[11], s14, 0x895cd7be);

		FF_SIMD(a1, b1, c1, d1, x_vec1[12], s11, 0x6b901122);
		FF_SIMD(a2, b2, c2, d2, x_vec2[12], s11, 0x6b901122);

		FF_SIMD(d1, a1, b1, c1, x_vec1[13], s12, 0xfd987193);
		FF_SIMD(d2, a2, b2, c2, x_vec2[13], s12, 0xfd987193);

		FF_SIMD(c1, d1, a1, b1, x_vec1[14], s13, 0xa679438e);
		FF_SIMD(c2, d2, a2, b2, x_vec2[14], s13, 0xa679438e);

		FF_SIMD(b1, c1, d1, a1, x_vec1[15], s14, 0x49b40821);
		FF_SIMD(b2, c2, d2, a2, x_vec2[15], s14, 0x49b40821);

		// 第二轮 (Round 2) - 交织执行
		GG_SIMD(a1, b1, c1, d1, x_vec1[1], s21, 0xf61e2562);
		GG_SIMD(a2, b2, c2, d2, x_vec2[1], s21, 0xf61e2562);

		GG_SIMD(d1, a1, b1, c1, x_vec1[6], s22, 0xc040b340);
		GG_SIMD(d2, a2, b2, c2, x_vec2[6], s22, 0xc040b340);

		GG_SIMD(c1, d1, a1, b1, x_vec1[11], s23, 0x265e5a51);
		GG_SIMD(c2, d2, a2, b2, x_vec2[11], s23, 0x265e5a51);

		GG_SIMD(b1, c1, d1, a1, x_vec1[0], s24, 0xe9b6c7aa);
		GG_SIMD(b2, c2, d2, a2, x_vec2[0], s24, 0xe9b6c7aa);

		GG_SIMD(a1, b1, c1, d1, x_vec1[5], s21, 0xd62f105d);
		GG_SIMD(a2, b2, c2, d2, x_vec2[5], s21, 0xd62f105d);

		GG_SIMD(d1, a1, b1, c1, x_vec1[10], s22, 0x02441453);
		GG_SIMD(d2, a2, b2, c2, x_vec2[10], s22, 0x02441453);

		GG_SIMD(c1, d1, a1, b1, x_vec1[15], s23, 0xd8a1e681);
		GG_SIMD(c2, d2, a2, b2, x_vec2[15], s23, 0xd8a1e681);

		GG_SIMD(b1, c1, d1, a1, x_vec1[4], s24, 0xe7d3fbc8);
		GG_SIMD(b2, c2, d2, a2, x_vec2[4], s24, 0xe7d3fbc8);

		GG_SIMD(a1, b1, c1, d1, x_vec1[9], s21, 0x21e1cde6);
		GG_SIMD(a2, b2, c2, d2, x_vec2[9], s21, 0x21e1cde6);

		GG_SIMD(d1, a1, b1, c1, x_vec1[14], s22, 0xc33707d6);
		GG_SIMD(d2, a2, b2, c2, x_vec2[14], s22, 0xc33707d6);

		GG_SIMD(c1, d1, a1, b1, x_vec1[3], s23, 0xf4d50d87);
		GG_SIMD(c2, d2, a2, b2, x_vec2[3], s23, 0xf4d50d87);

		GG_SIMD(b1, c1, d1, a1, x_vec1[8], s24, 0x455a14ed);
		GG_SIMD(b2, c2, d2, a2, x_vec2[8], s24, 0x455a14ed);

		GG_SIMD(a1, b1, c1, d1, x_vec1[13], s21, 0xa9e3e905);
		GG_SIMD(a2, b2, c2, d2, x_vec2[13], s21, 0xa9e3e905);

		GG_SIMD(d1, a1, b1, c1, x_vec1[2], s22, 0xfcefa3f8);
		GG_SIMD(d2, a2, b2, c2, x_vec2[2], s22, 0xfcefa3f8);

		GG_SIMD(c1, d1, a1, b1, x_vec1[7], s23, 0x676f02d9);
		GG_SIMD(c2, d2, a2, b2, x_vec2[7], s23, 0x676f02d9);

		GG_SIMD(b1, c1, d1, a1, x_vec1[12], s24, 0x8d2a4c8a);
		GG_SIMD(b2, c2, d2, a2, x_vec2[12], s24, 0x8d2a4c8a);

		// 第三轮 (Round 3) - 交织执行
		HH_SIMD(a1, b1, c1, d1, x_vec1[5], s31, 0xfffa3942);
		HH_SIMD(a2, b2, c2, d2, x_vec2[5], s31, 0xfffa3942);

		HH_SIMD(d1, a1, b1, c1, x_vec1[8], s32, 0x8771f681);
		HH_SIMD(d2, a2, b2, c2, x_vec2[8], s32, 0x8771f681);

		HH_SIMD(c1, d1, a1, b1, x_vec1[11], s33, 0x6d9d6122);
		HH_SIMD(c2, d2, a2, b2, x_vec2[11], s33, 0x6d9d6122);

		HH_SIMD(b1, c1, d1, a1, x_vec1[14], s34, 0xfde5380c);
		HH_SIMD(b2, c2, d2, a2, x_vec2[14], s34, 0xfde5380c);

		HH_SIMD(a1, b1, c1, d1, x_vec1[1], s31, 0xa4beea44);
		HH_SIMD(a2, b2, c2, d2, x_vec2[1], s31, 0xa4beea44);

		HH_SIMD(d1, a1, b1, c1, x_vec1[4], s32, 0x4bdecfa9);
		HH_SIMD(d2, a2, b2, c2, x_vec2[4], s32, 0x4bdecfa9);

		HH_SIMD(c1, d1, a1, b1, x_vec1[7], s33, 0xf6bb4b60);
		HH_SIMD(c2, d2, a2, b2, x_vec2[7], s33, 0xf6bb4b60);

		HH_SIMD(b1, c1, d1, a1, x_vec1[10], s34, 0xbebfbc70);
		HH_SIMD(b2, c2, d2, a2, x_vec2[10], s34, 0xbebfbc70);

		HH_SIMD(a1, b1, c1, d1, x_vec1[13], s31, 0x289b7ec6);
		HH_SIMD(a2, b2, c2, d2, x_vec2[13], s31, 0x289b7ec6);

		HH_SIMD(d1, a1, b1, c1, x_vec1[0], s32, 0xeaa127fa);
		HH_SIMD(d2, a2, b2, c2, x_vec2[0], s32, 0xeaa127fa);

		HH_SIMD(c1, d1, a1, b1, x_vec1[3], s33, 0xd4ef3085);
		HH_SIMD(c2, d2, a2, b2, x_vec2[3], s33, 0xd4ef3085);

		HH_SIMD(b1, c1, d1, a1, x_vec1[6], s34, 0x04881d05);
		HH_SIMD(b2, c2, d2, a2, x_vec2[6], s34, 0x04881d05);

		HH_SIMD(a1, b1, c1, d1, x_vec1[9], s31, 0xd9d4d039);
		HH_SIMD(a2, b2, c2, d2, x_vec2[9], s31, 0xd9d4d039);

		HH_SIMD(d1, a1, b1, c1, x_vec1[12], s32, 0xe6db99e5);
		HH_SIMD(d2, a2, b2, c2, x_vec2[12], s32, 0xe6db99e5);

		HH_SIMD(c1, d1, a1, b1, x_vec1[15], s33, 0x1fa27cf8);
		HH_SIMD(c2, d2, a2, b2, x_vec2[15], s33, 0x1fa27cf8);

		HH_SIMD(b1, c1, d1, a1, x_vec1[2], s34, 0xc4ac5665);
		HH_SIMD(b2, c2, d2, a2, x_vec2[2], s34, 0xc4ac5665);

		// 第四轮 (Round 4) - 交织执行
		II_SIMD(a1, b1, c1, d1, x_vec1[0], s41, 0xf4292244);
		II_SIMD(a2, b2, c2, d2, x_vec2[0], s41, 0xf4292244);

		II_SIMD(d1, a1, b1, c1, x_vec1[7], s42, 0x432aff97);
		II_SIMD(d2, a2, b2, c2, x_vec2[7], s42, 0x432aff97);

		II_SIMD(c1, d1, a1, b1, x_vec1[14], s43, 0xab9423a7);
		II_SIMD(c2, d2, a2, b2, x_vec2[14], s43, 0xab9423a7);

		II_SIMD(b1, c1, d1, a1, x_vec1[5], s44, 0xfc93a039);
		II_SIMD(b2, c2, d2, a2, x_vec2[5], s44, 0xfc93a039);

		II_SIMD(a1, b1, c1, d1, x_vec1[12], s41, 0x655b59c3);
		II_SIMD(a2, b2, c2, d2, x_vec2[12], s41, 0x655b59c3);

		II_SIMD(d1, a1, b1, c1, x_vec1[3], s42, 0x8f0ccc92);
		II_SIMD(d2, a2, b2, c2, x_vec2[3], s42, 0x8f0ccc92);

		II_SIMD(c1, d1, a1, b1, x_vec1[10], s43, 0xffeff47d);
		II_SIMD(c2, d2, a2, b2, x_vec2[10], s43, 0xffeff47d);

		II_SIMD(b1, c1, d1, a1, x_vec1[1], s44, 0x85845dd1);
		II_SIMD(b2, c2, d2, a2, x_vec2[1], s44, 0x85845dd1);

		II_SIMD(a1, b1, c1, d1, x_vec1[8], s41, 0x6fa87e4f);
		II_SIMD(a2, b2, c2, d2, x_vec2[8], s41, 0x6fa87e4f);

		II_SIMD(d1, a1, b1, c1, x_vec1[15], s42, 0xfe2ce6e0);
		II_SIMD(d2, a2, b2, c2, x_vec2[15], s42, 0xfe2ce6e0);

		II_SIMD(c1, d1, a1, b1, x_vec1[6], s43, 0xa3014314);
		II_SIMD(c2, d2, a2, b2, x_vec2[6], s43, 0xa3014314);

		II_SIMD(b1, c1, d1, a1, x_vec1[13], s44, 0x4e0811a1);
		II_SIMD(b2, c2, d2, a2, x_vec2[13], s44, 0x4e0811a1);

		II_SIMD(a1, b1, c1, d1, x_vec1[4], s41, 0xf7537e82);
		II_SIMD(a2, b2, c2, d2, x_vec2[4], s41, 0xf7537e82);

		II_SIMD(d1, a1, b1, c1, x_vec1[11], s42, 0xbd3af235);
		II_SIMD(d2, a2, b2, c2, x_vec2[11], s42, 0xbd3af235);

		II_SIMD(c1, d1, a1, b1, x_vec1[2], s43, 0x2ad7d2bb);
		II_SIMD(c2, d2, a2, b2, x_vec2[2], s43, 0x2ad7d2bb);

		II_SIMD(b1, c1, d1, a1, x_vec1[9], s44, 0xeb86d391);
		II_SIMD(b2, c2, d2, a2, x_vec2[9], s44, 0xeb86d391);

		// 继续所有轮次的交织执行...
		// 这里省略了大量类似代码，实际实现需要完整列出所有FF/GG/HH/II操作

		// 最后更新状态值
		a1 = vaddq_u32(a1, aa1);
		a2 = vaddq_u32(a2, aa2);
		b1 = vaddq_u32(b1, bb1);
		b2 = vaddq_u32(b2, bb2);
		c1 = vaddq_u32(c1, cc1);
		c2 = vaddq_u32(c2, cc2);
		d1 = vaddq_u32(d1, dd1);
		d2 = vaddq_u32(d2, dd2);

		// 更新保存的初始值，为下一个块做准备
		aa1 = a1;
		bb1 = b1;
		cc1 = c1;
		dd1 = d1;
		aa2 = a2;
		bb2 = b2;
		cc2 = c2;
		dd2 = d2;
	}

	// 存储结果
	uint32_t result_a1[4], result_b1[4], result_c1[4], result_d1[4];
	uint32_t result_a2[4], result_b2[4], result_c2[4], result_d2[4];

	vst1q_u32(result_a1, a1);
	vst1q_u32(result_b1, b1);
	vst1q_u32(result_c1, c1);
	vst1q_u32(result_d1, d1);

	vst1q_u32(result_a2, a2);
	vst1q_u32(result_b2, b2);
	vst1q_u32(result_c2, c2);
	vst1q_u32(result_d2, d2);

	// 字节序转换并复制到结果数组
	for (int i = 0; i < 4; i++)
	{
		// 第一组的结果
		states[i][0] = ((result_a1[i] & 0xFF) << 24) |
					   ((result_a1[i] & 0xFF00) << 8) |
					   ((result_a1[i] & 0xFF0000) >> 8) |
					   ((result_a1[i] & 0xFF000000) >> 24);
		states[i][1] = ((result_b1[i] & 0xFF) << 24) |
					   ((result_b1[i] & 0xFF00) << 8) |
					   ((result_b1[i] & 0xFF0000) >> 8) |
					   ((result_b1[i] & 0xFF000000) >> 24);
		states[i][2] = ((result_c1[i] & 0xFF) << 24) |
					   ((result_c1[i] & 0xFF00) << 8) |
					   ((result_c1[i] & 0xFF0000) >> 8) |
					   ((result_c1[i] & 0xFF000000) >> 24);
		states[i][3] = ((result_d1[i] & 0xFF) << 24) |
					   ((result_d1[i] & 0xFF00) << 8) |
					   ((result_d1[i] & 0xFF0000) >> 8) |
					   ((result_d1[i] & 0xFF000000) >> 24);

		// 第二组的结果
		states[i + 4][0] = ((result_a2[i] & 0xFF) << 24) |
						   ((result_a2[i] & 0xFF00) << 8) |
						   ((result_a2[i] & 0xFF0000) >> 8) |
						   ((result_a2[i] & 0xFF000000) >> 24);
		states[i + 4][1] = ((result_b2[i] & 0xFF) << 24) |
						   ((result_b2[i] & 0xFF00) << 8) |
						   ((result_b2[i] & 0xFF0000) >> 8) |
						   ((result_b2[i] & 0xFF000000) >> 24);
		states[i + 4][2] = ((result_c2[i] & 0xFF) << 24) |
						   ((result_c2[i] & 0xFF00) << 8) |
						   ((result_c2[i] & 0xFF0000) >> 8) |
						   ((result_c2[i] & 0xFF000000) >> 24);
		states[i + 4][3] = ((result_d2[i] & 0xFF) << 24) |
						   ((result_d2[i] & 0xFF00) << 8) |
						   ((result_d2[i] & 0xFF0000) >> 8) |
						   ((result_d2[i] & 0xFF000000) >> 24);
	}

	// 释放内存
	for (int i = 0; i < 8; i++)
	{
		delete[] paddedMessages[i];
	}
}


void MD5Hash_SIMD12(const string input[12], bit32 states[12][4])
{
    // 为12个输入准备缓冲区
    Byte *paddedMessages[12];
    int paddedLengths[12];
    int n_blocks[12];

    // 处理每个字符串，初始化状态
    for (int i = 0; i < 12; i++)
    {
        paddedMessages[i] = StringProcess(input[i], &paddedLengths[i]);
        n_blocks[i] = paddedLengths[i] / 64;
        states[i][0] = 0x67452301;
        states[i][1] = 0xefcdab89;
        states[i][2] = 0x98badcfe;
        states[i][3] = 0x10325476;
    }

    // 获取最大块数
    int max_blocks = *max_element(n_blocks, n_blocks + 12);

    // 创建三组4个字符串的状态变量
    // 第一组
    uint32x4_t a1 = vdupq_n_u32(0x67452301);
    uint32x4_t b1 = vdupq_n_u32(0xefcdab89);
    uint32x4_t c1 = vdupq_n_u32(0x98badcfe);
    uint32x4_t d1 = vdupq_n_u32(0x10325476);
    
    // 第二组
    uint32x4_t a2 = vdupq_n_u32(0x67452301);
    uint32x4_t b2 = vdupq_n_u32(0xefcdab89);
    uint32x4_t c2 = vdupq_n_u32(0x98badcfe);
    uint32x4_t d2 = vdupq_n_u32(0x10325476);
    
    // 第三组
    uint32x4_t a3 = vdupq_n_u32(0x67452301);
    uint32x4_t b3 = vdupq_n_u32(0xefcdab89);
    uint32x4_t c3 = vdupq_n_u32(0x98badcfe);
    uint32x4_t d3 = vdupq_n_u32(0x10325476);

    // 保存初始状态
    uint32x4_t aa1 = a1, bb1 = b1, cc1 = c1, dd1 = d1;
    uint32x4_t aa2 = a2, bb2 = b2, cc2 = c2, dd2 = d2;
    uint32x4_t aa3 = a3, bb3 = b3, cc3 = c3, dd3 = d3;

    // 临时缓冲区
    uint32x4_t x_vec1[16], x_vec2[16], x_vec3[16];
    bit32 x1[4], x2[4], x3[4];

    // 处理每个块
    for (int i = 0; i < max_blocks; ++i)
    {
        // 重置状态为保存的初始值
        a1 = aa1; b1 = bb1; c1 = cc1; d1 = dd1;
        a2 = aa2; b2 = bb2; c2 = cc2; d2 = dd2;
        a3 = aa3; b3 = bb3; c3 = cc3; d3 = dd3;

        // 为三批4个字符串分别加载消息块
        for (int j = 0; j < 16; j++)
        {
            int msg_offset = i * 64 + j * 4;

            // 处理第一组4个字符串
            for (int k = 0; k < 4; k++)
            {
                if (i < n_blocks[k])
                {
                    x1[k] = (paddedMessages[k][msg_offset]) |
                           (paddedMessages[k][msg_offset + 1] << 8) |
                           (paddedMessages[k][msg_offset + 2] << 16) |
                           (paddedMessages[k][msg_offset + 3] << 24);
                }
                else
                {
                    x1[k] = 0;
                }
            }

            // 处理第二组4个字符串
            for (int k = 0; k < 4; k++)
            {
                if (i < n_blocks[k+4])
                {
                    x2[k] = (paddedMessages[k+4][msg_offset]) |
                           (paddedMessages[k+4][msg_offset + 1] << 8) |
                           (paddedMessages[k+4][msg_offset + 2] << 16) |
                           (paddedMessages[k+4][msg_offset + 3] << 24);
                }
                else
                {
                    x2[k] = 0;
                }
            }

            // 处理第三组4个字符串
            for (int k = 0; k < 4; k++)
            {
                if (i < n_blocks[k+8])
                {
                    x3[k] = (paddedMessages[k+8][msg_offset]) |
                           (paddedMessages[k+8][msg_offset + 1] << 8) |
                           (paddedMessages[k+8][msg_offset + 2] << 16) |
                           (paddedMessages[k+8][msg_offset + 3] << 24);
                }
                else
                {
                    x3[k] = 0;
                }
            }

            x_vec1[j] = vld1q_u32(x1);
            x_vec2[j] = vld1q_u32(x2);
            x_vec3[j] = vld1q_u32(x3);
        }

        // 第一轮 (Round 1) - 交织执行三组
        FF_SIMD(a1, b1, c1, d1, x_vec1[0], s11, 0xd76aa478);
        FF_SIMD(a2, b2, c2, d2, x_vec2[0], s11, 0xd76aa478);
        FF_SIMD(a3, b3, c3, d3, x_vec3[0], s11, 0xd76aa478);

        FF_SIMD(d1, a1, b1, c1, x_vec1[1], s12, 0xe8c7b756);
        FF_SIMD(d2, a2, b2, c2, x_vec2[1], s12, 0xe8c7b756);
        FF_SIMD(d3, a3, b3, c3, x_vec3[1], s12, 0xe8c7b756);

        FF_SIMD(c1, d1, a1, b1, x_vec1[2], s13, 0x242070db);
        FF_SIMD(c2, d2, a2, b2, x_vec2[2], s13, 0x242070db);
        FF_SIMD(c3, d3, a3, b3, x_vec3[2], s13, 0x242070db);

        FF_SIMD(b1, c1, d1, a1, x_vec1[3], s14, 0xc1bdceee);
        FF_SIMD(b2, c2, d2, a2, x_vec2[3], s14, 0xc1bdceee);
        FF_SIMD(b3, c3, d3, a3, x_vec3[3], s14, 0xc1bdceee);

        FF_SIMD(a1, b1, c1, d1, x_vec1[4], s11, 0xf57c0faf);
        FF_SIMD(a2, b2, c2, d2, x_vec2[4], s11, 0xf57c0faf);
        FF_SIMD(a3, b3, c3, d3, x_vec3[4], s11, 0xf57c0faf);

        FF_SIMD(d1, a1, b1, c1, x_vec1[5], s12, 0x4787c62a);
        FF_SIMD(d2, a2, b2, c2, x_vec2[5], s12, 0x4787c62a);
        FF_SIMD(d3, a3, b3, c3, x_vec3[5], s12, 0x4787c62a);

        FF_SIMD(c1, d1, a1, b1, x_vec1[6], s13, 0xa8304613);
        FF_SIMD(c2, d2, a2, b2, x_vec2[6], s13, 0xa8304613);
        FF_SIMD(c3, d3, a3, b3, x_vec3[6], s13, 0xa8304613);

        FF_SIMD(b1, c1, d1, a1, x_vec1[7], s14, 0xfd469501);
        FF_SIMD(b2, c2, d2, a2, x_vec2[7], s14, 0xfd469501);
        FF_SIMD(b3, c3, d3, a3, x_vec3[7], s14, 0xfd469501);

        FF_SIMD(a1, b1, c1, d1, x_vec1[8], s11, 0x698098d8);
        FF_SIMD(a2, b2, c2, d2, x_vec2[8], s11, 0x698098d8);
        FF_SIMD(a3, b3, c3, d3, x_vec3[8], s11, 0x698098d8);

        FF_SIMD(d1, a1, b1, c1, x_vec1[9], s12, 0x8b44f7af);
        FF_SIMD(d2, a2, b2, c2, x_vec2[9], s12, 0x8b44f7af);
        FF_SIMD(d3, a3, b3, c3, x_vec3[9], s12, 0x8b44f7af);

        FF_SIMD(c1, d1, a1, b1, x_vec1[10], s13, 0xffff5bb1);
        FF_SIMD(c2, d2, a2, b2, x_vec2[10], s13, 0xffff5bb1);
        FF_SIMD(c3, d3, a3, b3, x_vec3[10], s13, 0xffff5bb1);

        FF_SIMD(b1, c1, d1, a1, x_vec1[11], s14, 0x895cd7be);
        FF_SIMD(b2, c2, d2, a2, x_vec2[11], s14, 0x895cd7be);
        FF_SIMD(b3, c3, d3, a3, x_vec3[11], s14, 0x895cd7be);

        FF_SIMD(a1, b1, c1, d1, x_vec1[12], s11, 0x6b901122);
        FF_SIMD(a2, b2, c2, d2, x_vec2[12], s11, 0x6b901122);
        FF_SIMD(a3, b3, c3, d3, x_vec3[12], s11, 0x6b901122);

        FF_SIMD(d1, a1, b1, c1, x_vec1[13], s12, 0xfd987193);
        FF_SIMD(d2, a2, b2, c2, x_vec2[13], s12, 0xfd987193);
        FF_SIMD(d3, a3, b3, c3, x_vec3[13], s12, 0xfd987193);

        FF_SIMD(c1, d1, a1, b1, x_vec1[14], s13, 0xa679438e);
        FF_SIMD(c2, d2, a2, b2, x_vec2[14], s13, 0xa679438e);
        FF_SIMD(c3, d3, a3, b3, x_vec3[14], s13, 0xa679438e);

        FF_SIMD(b1, c1, d1, a1, x_vec1[15], s14, 0x49b40821);
        FF_SIMD(b2, c2, d2, a2, x_vec2[15], s14, 0x49b40821);
        FF_SIMD(b3, c3, d3, a3, x_vec3[15], s14, 0x49b40821);

        // 第二轮 (Round 2) - 交织执行三组
        GG_SIMD(a1, b1, c1, d1, x_vec1[1], s21, 0xf61e2562);
        GG_SIMD(a2, b2, c2, d2, x_vec2[1], s21, 0xf61e2562);
        GG_SIMD(a3, b3, c3, d3, x_vec3[1], s21, 0xf61e2562);

        GG_SIMD(d1, a1, b1, c1, x_vec1[6], s22, 0xc040b340);
        GG_SIMD(d2, a2, b2, c2, x_vec2[6], s22, 0xc040b340);
        GG_SIMD(d3, a3, b3, c3, x_vec3[6], s22, 0xc040b340);

        GG_SIMD(c1, d1, a1, b1, x_vec1[11], s23, 0x265e5a51);
        GG_SIMD(c2, d2, a2, b2, x_vec2[11], s23, 0x265e5a51);
        GG_SIMD(c3, d3, a3, b3, x_vec3[11], s23, 0x265e5a51);

        GG_SIMD(b1, c1, d1, a1, x_vec1[0], s24, 0xe9b6c7aa);
        GG_SIMD(b2, c2, d2, a2, x_vec2[0], s24, 0xe9b6c7aa);
        GG_SIMD(b3, c3, d3, a3, x_vec3[0], s24, 0xe9b6c7aa);

        GG_SIMD(a1, b1, c1, d1, x_vec1[5], s21, 0xd62f105d);
        GG_SIMD(a2, b2, c2, d2, x_vec2[5], s21, 0xd62f105d);
        GG_SIMD(a3, b3, c3, d3, x_vec3[5], s21, 0xd62f105d);

        GG_SIMD(d1, a1, b1, c1, x_vec1[10], s22, 0x02441453);
        GG_SIMD(d2, a2, b2, c2, x_vec2[10], s22, 0x02441453);
        GG_SIMD(d3, a3, b3, c3, x_vec3[10], s22, 0x02441453);

        GG_SIMD(c1, d1, a1, b1, x_vec1[15], s23, 0xd8a1e681);
        GG_SIMD(c2, d2, a2, b2, x_vec2[15], s23, 0xd8a1e681);
        GG_SIMD(c3, d3, a3, b3, x_vec3[15], s23, 0xd8a1e681);

        GG_SIMD(b1, c1, d1, a1, x_vec1[4], s24, 0xe7d3fbc8);
        GG_SIMD(b2, c2, d2, a2, x_vec2[4], s24, 0xe7d3fbc8);
        GG_SIMD(b3, c3, d3, a3, x_vec3[4], s24, 0xe7d3fbc8);

        GG_SIMD(a1, b1, c1, d1, x_vec1[9], s21, 0x21e1cde6);
        GG_SIMD(a2, b2, c2, d2, x_vec2[9], s21, 0x21e1cde6);
        GG_SIMD(a3, b3, c3, d3, x_vec3[9], s21, 0x21e1cde6);

        GG_SIMD(d1, a1, b1, c1, x_vec1[14], s22, 0xc33707d6);
        GG_SIMD(d2, a2, b2, c2, x_vec2[14], s22, 0xc33707d6);
        GG_SIMD(d3, a3, b3, c3, x_vec3[14], s22, 0xc33707d6);

        GG_SIMD(c1, d1, a1, b1, x_vec1[3], s23, 0xf4d50d87);
        GG_SIMD(c2, d2, a2, b2, x_vec2[3], s23, 0xf4d50d87);
        GG_SIMD(c3, d3, a3, b3, x_vec3[3], s23, 0xf4d50d87);

        GG_SIMD(b1, c1, d1, a1, x_vec1[8], s24, 0x455a14ed);
        GG_SIMD(b2, c2, d2, a2, x_vec2[8], s24, 0x455a14ed);
        GG_SIMD(b3, c3, d3, a3, x_vec3[8], s24, 0x455a14ed);

        GG_SIMD(a1, b1, c1, d1, x_vec1[13], s21, 0xa9e3e905);
        GG_SIMD(a2, b2, c2, d2, x_vec2[13], s21, 0xa9e3e905);
        GG_SIMD(a3, b3, c3, d3, x_vec3[13], s21, 0xa9e3e905);

        GG_SIMD(d1, a1, b1, c1, x_vec1[2], s22, 0xfcefa3f8);
        GG_SIMD(d2, a2, b2, c2, x_vec2[2], s22, 0xfcefa3f8);
        GG_SIMD(d3, a3, b3, c3, x_vec3[2], s22, 0xfcefa3f8);

        GG_SIMD(c1, d1, a1, b1, x_vec1[7], s23, 0x676f02d9);
        GG_SIMD(c2, d2, a2, b2, x_vec2[7], s23, 0x676f02d9);
        GG_SIMD(c3, d3, a3, b3, x_vec3[7], s23, 0x676f02d9);

        GG_SIMD(b1, c1, d1, a1, x_vec1[12], s24, 0x8d2a4c8a);
        GG_SIMD(b2, c2, d2, a2, x_vec2[12], s24, 0x8d2a4c8a);
        GG_SIMD(b3, c3, d3, a3, x_vec3[12], s24, 0x8d2a4c8a);

        // 第三轮 (Round 3) - 交织执行三组
        HH_SIMD(a1, b1, c1, d1, x_vec1[5], s31, 0xfffa3942);
        HH_SIMD(a2, b2, c2, d2, x_vec2[5], s31, 0xfffa3942);
        HH_SIMD(a3, b3, c3, d3, x_vec3[5], s31, 0xfffa3942);

        HH_SIMD(d1, a1, b1, c1, x_vec1[8], s32, 0x8771f681);
        HH_SIMD(d2, a2, b2, c2, x_vec2[8], s32, 0x8771f681);
        HH_SIMD(d3, a3, b3, c3, x_vec3[8], s32, 0x8771f681);

        HH_SIMD(c1, d1, a1, b1, x_vec1[11], s33, 0x6d9d6122);
        HH_SIMD(c2, d2, a2, b2, x_vec2[11], s33, 0x6d9d6122);
        HH_SIMD(c3, d3, a3, b3, x_vec3[11], s33, 0x6d9d6122);

        HH_SIMD(b1, c1, d1, a1, x_vec1[14], s34, 0xfde5380c);
        HH_SIMD(b2, c2, d2, a2, x_vec2[14], s34, 0xfde5380c);
        HH_SIMD(b3, c3, d3, a3, x_vec3[14], s34, 0xfde5380c);

        HH_SIMD(a1, b1, c1, d1, x_vec1[1], s31, 0xa4beea44);
        HH_SIMD(a2, b2, c2, d2, x_vec2[1], s31, 0xa4beea44);
        HH_SIMD(a3, b3, c3, d3, x_vec3[1], s31, 0xa4beea44);

        HH_SIMD(d1, a1, b1, c1, x_vec1[4], s32, 0x4bdecfa9);
        HH_SIMD(d2, a2, b2, c2, x_vec2[4], s32, 0x4bdecfa9);
        HH_SIMD(d3, a3, b3, c3, x_vec3[4], s32, 0x4bdecfa9);

        HH_SIMD(c1, d1, a1, b1, x_vec1[7], s33, 0xf6bb4b60);
        HH_SIMD(c2, d2, a2, b2, x_vec2[7], s33, 0xf6bb4b60);
        HH_SIMD(c3, d3, a3, b3, x_vec3[7], s33, 0xf6bb4b60);

        HH_SIMD(b1, c1, d1, a1, x_vec1[10], s34, 0xbebfbc70);
        HH_SIMD(b2, c2, d2, a2, x_vec2[10], s34, 0xbebfbc70);
        HH_SIMD(b3, c3, d3, a3, x_vec3[10], s34, 0xbebfbc70);

        HH_SIMD(a1, b1, c1, d1, x_vec1[13], s31, 0x289b7ec6);
        HH_SIMD(a2, b2, c2, d2, x_vec2[13], s31, 0x289b7ec6);
        HH_SIMD(a3, b3, c3, d3, x_vec3[13], s31, 0x289b7ec6);

        HH_SIMD(d1, a1, b1, c1, x_vec1[0], s32, 0xeaa127fa);
        HH_SIMD(d2, a2, b2, c2, x_vec2[0], s32, 0xeaa127fa);
        HH_SIMD(d3, a3, b3, c3, x_vec3[0], s32, 0xeaa127fa);

        HH_SIMD(c1, d1, a1, b1, x_vec1[3], s33, 0xd4ef3085);
        HH_SIMD(c2, d2, a2, b2, x_vec2[3], s33, 0xd4ef3085);
        HH_SIMD(c3, d3, a3, b3, x_vec3[3], s33, 0xd4ef3085);

        HH_SIMD(b1, c1, d1, a1, x_vec1[6], s34, 0x04881d05);
        HH_SIMD(b2, c2, d2, a2, x_vec2[6], s34, 0x04881d05);
        HH_SIMD(b3, c3, d3, a3, x_vec3[6], s34, 0x04881d05);

        HH_SIMD(a1, b1, c1, d1, x_vec1[9], s31, 0xd9d4d039);
        HH_SIMD(a2, b2, c2, d2, x_vec2[9], s31, 0xd9d4d039);
        HH_SIMD(a3, b3, c3, d3, x_vec3[9], s31, 0xd9d4d039);

        HH_SIMD(d1, a1, b1, c1, x_vec1[12], s32, 0xe6db99e5);
        HH_SIMD(d2, a2, b2, c2, x_vec2[12], s32, 0xe6db99e5);
        HH_SIMD(d3, a3, b3, c3, x_vec3[12], s32, 0xe6db99e5);

        HH_SIMD(c1, d1, a1, b1, x_vec1[15], s33, 0x1fa27cf8);
        HH_SIMD(c2, d2, a2, b2, x_vec2[15], s33, 0x1fa27cf8);
        HH_SIMD(c3, d3, a3, b3, x_vec3[15], s33, 0x1fa27cf8);

        HH_SIMD(b1, c1, d1, a1, x_vec1[2], s34, 0xc4ac5665);
        HH_SIMD(b2, c2, d2, a2, x_vec2[2], s34, 0xc4ac5665);
        HH_SIMD(b3, c3, d3, a3, x_vec3[2], s34, 0xc4ac5665);

        // 第四轮 (Round 4) - 交织执行三组
        II_SIMD(a1, b1, c1, d1, x_vec1[0], s41, 0xf4292244);
        II_SIMD(a2, b2, c2, d2, x_vec2[0], s41, 0xf4292244);
        II_SIMD(a3, b3, c3, d3, x_vec3[0], s41, 0xf4292244);

        II_SIMD(d1, a1, b1, c1, x_vec1[7], s42, 0x432aff97);
        II_SIMD(d2, a2, b2, c2, x_vec2[7], s42, 0x432aff97);
        II_SIMD(d3, a3, b3, c3, x_vec3[7], s42, 0x432aff97);

        II_SIMD(c1, d1, a1, b1, x_vec1[14], s43, 0xab9423a7);
        II_SIMD(c2, d2, a2, b2, x_vec2[14], s43, 0xab9423a7);
        II_SIMD(c3, d3, a3, b3, x_vec3[14], s43, 0xab9423a7);

        II_SIMD(b1, c1, d1, a1, x_vec1[5], s44, 0xfc93a039);
        II_SIMD(b2, c2, d2, a2, x_vec2[5], s44, 0xfc93a039);
        II_SIMD(b3, c3, d3, a3, x_vec3[5], s44, 0xfc93a039);

        II_SIMD(a1, b1, c1, d1, x_vec1[12], s41, 0x655b59c3);
        II_SIMD(a2, b2, c2, d2, x_vec2[12], s41, 0x655b59c3);
        II_SIMD(a3, b3, c3, d3, x_vec3[12], s41, 0x655b59c3);

        II_SIMD(d1, a1, b1, c1, x_vec1[3], s42, 0x8f0ccc92);
        II_SIMD(d2, a2, b2, c2, x_vec2[3], s42, 0x8f0ccc92);
        II_SIMD(d3, a3, b3, c3, x_vec3[3], s42, 0x8f0ccc92);

        II_SIMD(c1, d1, a1, b1, x_vec1[10], s43, 0xffeff47d);
        II_SIMD(c2, d2, a2, b2, x_vec2[10], s43, 0xffeff47d);
        II_SIMD(c3, d3, a3, b3, x_vec3[10], s43, 0xffeff47d);

        II_SIMD(b1, c1, d1, a1, x_vec1[1], s44, 0x85845dd1);
        II_SIMD(b2, c2, d2, a2, x_vec2[1], s44, 0x85845dd1);
        II_SIMD(b3, c3, d3, a3, x_vec3[1], s44, 0x85845dd1);

        II_SIMD(a1, b1, c1, d1, x_vec1[8], s41, 0x6fa87e4f);
        II_SIMD(a2, b2, c2, d2, x_vec2[8], s41, 0x6fa87e4f);
        II_SIMD(a3, b3, c3, d3, x_vec3[8], s41, 0x6fa87e4f);

        II_SIMD(d1, a1, b1, c1, x_vec1[15], s42, 0xfe2ce6e0);
        II_SIMD(d2, a2, b2, c2, x_vec2[15], s42, 0xfe2ce6e0);
        II_SIMD(d3, a3, b3, c3, x_vec3[15], s42, 0xfe2ce6e0);

        II_SIMD(c1, d1, a1, b1, x_vec1[6], s43, 0xa3014314);
        II_SIMD(c2, d2, a2, b2, x_vec2[6], s43, 0xa3014314);
        II_SIMD(c3, d3, a3, b3, x_vec3[6], s43, 0xa3014314);

        II_SIMD(b1, c1, d1, a1, x_vec1[13], s44, 0x4e0811a1);
        II_SIMD(b2, c2, d2, a2, x_vec2[13], s44, 0x4e0811a1);
        II_SIMD(b3, c3, d3, a3, x_vec3[13], s44, 0x4e0811a1);

        II_SIMD(a1, b1, c1, d1, x_vec1[4], s41, 0xf7537e82);
        II_SIMD(a2, b2, c2, d2, x_vec2[4], s41, 0xf7537e82);
        II_SIMD(a3, b3, c3, d3, x_vec3[4], s41, 0xf7537e82);

        II_SIMD(d1, a1, b1, c1, x_vec1[11], s42, 0xbd3af235);
        II_SIMD(d2, a2, b2, c2, x_vec2[11], s42, 0xbd3af235);
        II_SIMD(d3, a3, b3, c3, x_vec3[11], s42, 0xbd3af235);

        II_SIMD(c1, d1, a1, b1, x_vec1[2], s43, 0x2ad7d2bb);
        II_SIMD(c2, d2, a2, b2, x_vec2[2], s43, 0x2ad7d2bb);
        II_SIMD(c3, d3, a3, b3, x_vec3[2], s43, 0x2ad7d2bb);

        II_SIMD(b1, c1, d1, a1, x_vec1[9], s44, 0xeb86d391);
        II_SIMD(b2, c2, d2, a2, x_vec2[9], s44, 0xeb86d391);
        II_SIMD(b3, c3, d3, a3, x_vec3[9], s44, 0xeb86d391);

        // 更新状态
        a1 = vaddq_u32(a1, aa1);
        b1 = vaddq_u32(b1, bb1);
        c1 = vaddq_u32(c1, cc1);
        d1 = vaddq_u32(d1, dd1);
        
        a2 = vaddq_u32(a2, aa2);
        b2 = vaddq_u32(b2, bb2);
        c2 = vaddq_u32(c2, cc2);
        d2 = vaddq_u32(d2, dd2);
        
        a3 = vaddq_u32(a3, aa3);
        b3 = vaddq_u32(b3, bb3);
        c3 = vaddq_u32(c3, cc3);
        d3 = vaddq_u32(d3, dd3);

        // 更新保存的初始值，为下一个块做准备
        aa1 = a1; bb1 = b1; cc1 = c1; dd1 = d1;
        aa2 = a2; bb2 = b2; cc2 = c2; dd2 = d2;
        aa3 = a3; bb3 = b3; cc3 = c3; dd3 = d3;
    }

    // 存储结果
    uint32_t result_a1[4], result_b1[4], result_c1[4], result_d1[4];
    uint32_t result_a2[4], result_b2[4], result_c2[4], result_d2[4];
    uint32_t result_a3[4], result_b3[4], result_c3[4], result_d3[4];
    
    vst1q_u32(result_a1, a1);
    vst1q_u32(result_b1, b1);
    vst1q_u32(result_c1, c1);
    vst1q_u32(result_d1, d1);
    
    vst1q_u32(result_a2, a2);
    vst1q_u32(result_b2, b2);
    vst1q_u32(result_c2, c2);
    vst1q_u32(result_d2, d2);
    
    vst1q_u32(result_a3, a3);
    vst1q_u32(result_b3, b3);
    vst1q_u32(result_c3, c3);
    vst1q_u32(result_d3, d3);
    
    // 字节序转换并复制到结果数组
    for (int i = 0; i < 4; i++) {
        // 第一组的结果
        states[i][0] = ((result_a1[i] & 0xFF) << 24) |
                      ((result_a1[i] & 0xFF00) << 8) |
                      ((result_a1[i] & 0xFF0000) >> 8) |
                      ((result_a1[i] & 0xFF000000) >> 24);
        states[i][1] = ((result_b1[i] & 0xFF) << 24) |
                      ((result_b1[i] & 0xFF00) << 8) |
                      ((result_b1[i] & 0xFF0000) >> 8) |
                      ((result_b1[i] & 0xFF000000) >> 24);
        states[i][2] = ((result_c1[i] & 0xFF) << 24) |
                      ((result_c1[i] & 0xFF00) << 8) |
                      ((result_c1[i] & 0xFF0000) >> 8) |
                      ((result_c1[i] & 0xFF000000) >> 24);
        states[i][3] = ((result_d1[i] & 0xFF) << 24) |
                      ((result_d1[i] & 0xFF00) << 8) |
                      ((result_d1[i] & 0xFF0000) >> 8) |
                      ((result_d1[i] & 0xFF000000) >> 24);
        
        // 第二组的结果
        states[i+4][0] = ((result_a2[i] & 0xFF) << 24) |
                        ((result_a2[i] & 0xFF00) << 8) |
                        ((result_a2[i] & 0xFF0000) >> 8) |
                        ((result_a2[i] & 0xFF000000) >> 24);
        states[i+4][1] = ((result_b2[i] & 0xFF) << 24) |
                        ((result_b2[i] & 0xFF00) << 8) |
                        ((result_b2[i] & 0xFF0000) >> 8) |
                        ((result_b2[i] & 0xFF000000) >> 24);
        states[i+4][2] = ((result_c2[i] & 0xFF) << 24) |
                        ((result_c2[i] & 0xFF00) << 8) |
                        ((result_c2[i] & 0xFF0000) >> 8) |
                        ((result_c2[i] & 0xFF000000) >> 24);
        states[i+4][3] = ((result_d2[i] & 0xFF) << 24) |
                        ((result_d2[i] & 0xFF00) << 8) |
                        ((result_d2[i] & 0xFF0000) >> 8) |
                        ((result_d2[i] & 0xFF000000) >> 24);
                        
        // 第三组的结果
        states[i+8][0] = ((result_a3[i] & 0xFF) << 24) |
                        ((result_a3[i] & 0xFF00) << 8) |
                        ((result_a3[i] & 0xFF0000) >> 8) |
                        ((result_a3[i] & 0xFF000000) >> 24);
        states[i+8][1] = ((result_b3[i] & 0xFF) << 24) |
                        ((result_b3[i] & 0xFF00) << 8) |
                        ((result_b3[i] & 0xFF0000) >> 8) |
                        ((result_b3[i] & 0xFF000000) >> 24);
        states[i+8][2] = ((result_c3[i] & 0xFF) << 24) |
                        ((result_c3[i] & 0xFF00) << 8) |
                        ((result_c3[i] & 0xFF0000) >> 8) |
                        ((result_c3[i] & 0xFF000000) >> 24);
        states[i+8][3] = ((result_d3[i] & 0xFF) << 24) |
                        ((result_d3[i] & 0xFF00) << 8) |
                        ((result_d3[i] & 0xFF0000) >> 8) |
                        ((result_d3[i] & 0xFF000000) >> 24);
    }
    
    // 释放内存
    for (int i = 0; i < 12; i++) {
        delete[] paddedMessages[i];
    }
}


void MD5Hash_SIMD16(const string input[16], bit32 states[16][4])
{
    // 为16个输入准备缓冲区
    Byte *paddedMessages[16];
    int paddedLengths[16];
    int n_blocks[16];

    // 处理每个字符串，初始化状态
    for (int i = 0; i < 16; i++)
    {
        paddedMessages[i] = StringProcess(input[i], &paddedLengths[i]);
        n_blocks[i] = paddedLengths[i] / 64;
        states[i][0] = 0x67452301;
        states[i][1] = 0xefcdab89;
        states[i][2] = 0x98badcfe;
        states[i][3] = 0x10325476;
    }

    // 获取最大块数
    int max_blocks = *max_element(n_blocks, n_blocks + 16);

    // 创建四组4个字符串的状态变量
    // 第一组
    uint32x4_t a1 = vdupq_n_u32(0x67452301);
    uint32x4_t b1 = vdupq_n_u32(0xefcdab89);
    uint32x4_t c1 = vdupq_n_u32(0x98badcfe);
    uint32x4_t d1 = vdupq_n_u32(0x10325476);
    
    // 第二组
    uint32x4_t a2 = vdupq_n_u32(0x67452301);
    uint32x4_t b2 = vdupq_n_u32(0xefcdab89);
    uint32x4_t c2 = vdupq_n_u32(0x98badcfe);
    uint32x4_t d2 = vdupq_n_u32(0x10325476);
    
    // 第三组
    uint32x4_t a3 = vdupq_n_u32(0x67452301);
    uint32x4_t b3 = vdupq_n_u32(0xefcdab89);
    uint32x4_t c3 = vdupq_n_u32(0x98badcfe);
    uint32x4_t d3 = vdupq_n_u32(0x10325476);
    
    // 第四组
    uint32x4_t a4 = vdupq_n_u32(0x67452301);
    uint32x4_t b4 = vdupq_n_u32(0xefcdab89);
    uint32x4_t c4 = vdupq_n_u32(0x98badcfe);
    uint32x4_t d4 = vdupq_n_u32(0x10325476);

    // 保存初始状态
    uint32x4_t aa1 = a1, bb1 = b1, cc1 = c1, dd1 = d1;
    uint32x4_t aa2 = a2, bb2 = b2, cc2 = c2, dd2 = d2;
    uint32x4_t aa3 = a3, bb3 = b3, cc3 = c3, dd3 = d3;
    uint32x4_t aa4 = a4, bb4 = b4, cc4 = c4, dd4 = d4;

    // 临时缓冲区
    uint32x4_t x_vec1[16], x_vec2[16], x_vec3[16], x_vec4[16];
    bit32 x1[4], x2[4], x3[4], x4[4];

    // 处理每个块
    for (int i = 0; i < max_blocks; ++i)
    {
        // 重置状态为保存的初始值
        a1 = aa1; b1 = bb1; c1 = cc1; d1 = dd1;
        a2 = aa2; b2 = bb2; c2 = cc2; d2 = dd2;
        a3 = aa3; b3 = bb3; c3 = cc3; d3 = dd3;
        a4 = aa4; b4 = bb4; c4 = cc4; d4 = dd4;

        // 为四批4个字符串分别加载消息块
        for (int j = 0; j < 16; j++)
        {
            int msg_offset = i * 64 + j * 4;

            // 处理第一组4个字符串
            for (int k = 0; k < 4; k++)
            {
                if (i < n_blocks[k])
                {
                    x1[k] = (paddedMessages[k][msg_offset]) |
                           (paddedMessages[k][msg_offset + 1] << 8) |
                           (paddedMessages[k][msg_offset + 2] << 16) |
                           (paddedMessages[k][msg_offset + 3] << 24);
                }
                else
                {
                    x1[k] = 0;
                }
            }

            // 处理第二组4个字符串
            for (int k = 0; k < 4; k++)
            {
                if (i < n_blocks[k+4])
                {
                    x2[k] = (paddedMessages[k+4][msg_offset]) |
                           (paddedMessages[k+4][msg_offset + 1] << 8) |
                           (paddedMessages[k+4][msg_offset + 2] << 16) |
                           (paddedMessages[k+4][msg_offset + 3] << 24);
                }
                else
                {
                    x2[k] = 0;
                }
            }

            // 处理第三组4个字符串
            for (int k = 0; k < 4; k++)
            {
                if (i < n_blocks[k+8])
                {
                    x3[k] = (paddedMessages[k+8][msg_offset]) |
                           (paddedMessages[k+8][msg_offset + 1] << 8) |
                           (paddedMessages[k+8][msg_offset + 2] << 16) |
                           (paddedMessages[k+8][msg_offset + 3] << 24);
                }
                else
                {
                    x3[k] = 0;
                }
            }
            
            // 处理第四组4个字符串
            for (int k = 0; k < 4; k++)
            {
                if (i < n_blocks[k+12])
                {
                    x4[k] = (paddedMessages[k+12][msg_offset]) |
                           (paddedMessages[k+12][msg_offset + 1] << 8) |
                           (paddedMessages[k+12][msg_offset + 2] << 16) |
                           (paddedMessages[k+12][msg_offset + 3] << 24);
                }
                else
                {
                    x4[k] = 0;
                }
            }

            x_vec1[j] = vld1q_u32(x1);
            x_vec2[j] = vld1q_u32(x2);
            x_vec3[j] = vld1q_u32(x3);
            x_vec4[j] = vld1q_u32(x4);
        }

        // 第一轮 (Round 1) - 交织执行四组
        FF_SIMD(a1, b1, c1, d1, x_vec1[0], s11, 0xd76aa478);
        FF_SIMD(a2, b2, c2, d2, x_vec2[0], s11, 0xd76aa478);
        FF_SIMD(a3, b3, c3, d3, x_vec3[0], s11, 0xd76aa478);
        FF_SIMD(a4, b4, c4, d4, x_vec4[0], s11, 0xd76aa478);

        FF_SIMD(d1, a1, b1, c1, x_vec1[1], s12, 0xe8c7b756);
        FF_SIMD(d2, a2, b2, c2, x_vec2[1], s12, 0xe8c7b756);
        FF_SIMD(d3, a3, b3, c3, x_vec3[1], s12, 0xe8c7b756);
        FF_SIMD(d4, a4, b4, c4, x_vec4[1], s12, 0xe8c7b756);

        FF_SIMD(c1, d1, a1, b1, x_vec1[2], s13, 0x242070db);
        FF_SIMD(c2, d2, a2, b2, x_vec2[2], s13, 0x242070db);
        FF_SIMD(c3, d3, a3, b3, x_vec3[2], s13, 0x242070db);
        FF_SIMD(c4, d4, a4, b4, x_vec4[2], s13, 0x242070db);

        FF_SIMD(b1, c1, d1, a1, x_vec1[3], s14, 0xc1bdceee);
        FF_SIMD(b2, c2, d2, a2, x_vec2[3], s14, 0xc1bdceee);
        FF_SIMD(b3, c3, d3, a3, x_vec3[3], s14, 0xc1bdceee);
        FF_SIMD(b4, c4, d4, a4, x_vec4[3], s14, 0xc1bdceee);

        FF_SIMD(a1, b1, c1, d1, x_vec1[4], s11, 0xf57c0faf);
        FF_SIMD(a2, b2, c2, d2, x_vec2[4], s11, 0xf57c0faf);
        FF_SIMD(a3, b3, c3, d3, x_vec3[4], s11, 0xf57c0faf);
        FF_SIMD(a4, b4, c4, d4, x_vec4[4], s11, 0xf57c0faf);

        FF_SIMD(d1, a1, b1, c1, x_vec1[5], s12, 0x4787c62a);
        FF_SIMD(d2, a2, b2, c2, x_vec2[5], s12, 0x4787c62a);
        FF_SIMD(d3, a3, b3, c3, x_vec3[5], s12, 0x4787c62a);
        FF_SIMD(d4, a4, b4, c4, x_vec4[5], s12, 0x4787c62a);

        FF_SIMD(c1, d1, a1, b1, x_vec1[6], s13, 0xa8304613);
        FF_SIMD(c2, d2, a2, b2, x_vec2[6], s13, 0xa8304613);
        FF_SIMD(c3, d3, a3, b3, x_vec3[6], s13, 0xa8304613);
        FF_SIMD(c4, d4, a4, b4, x_vec4[6], s13, 0xa8304613);

        FF_SIMD(b1, c1, d1, a1, x_vec1[7], s14, 0xfd469501);
        FF_SIMD(b2, c2, d2, a2, x_vec2[7], s14, 0xfd469501);
        FF_SIMD(b3, c3, d3, a3, x_vec3[7], s14, 0xfd469501);
        FF_SIMD(b4, c4, d4, a4, x_vec4[7], s14, 0xfd469501);

        FF_SIMD(a1, b1, c1, d1, x_vec1[8], s11, 0x698098d8);
        FF_SIMD(a2, b2, c2, d2, x_vec2[8], s11, 0x698098d8);
        FF_SIMD(a3, b3, c3, d3, x_vec3[8], s11, 0x698098d8);
        FF_SIMD(a4, b4, c4, d4, x_vec4[8], s11, 0x698098d8);

        FF_SIMD(d1, a1, b1, c1, x_vec1[9], s12, 0x8b44f7af);
        FF_SIMD(d2, a2, b2, c2, x_vec2[9], s12, 0x8b44f7af);
        FF_SIMD(d3, a3, b3, c3, x_vec3[9], s12, 0x8b44f7af);
        FF_SIMD(d4, a4, b4, c4, x_vec4[9], s12, 0x8b44f7af);

        FF_SIMD(c1, d1, a1, b1, x_vec1[10], s13, 0xffff5bb1);
        FF_SIMD(c2, d2, a2, b2, x_vec2[10], s13, 0xffff5bb1);
        FF_SIMD(c3, d3, a3, b3, x_vec3[10], s13, 0xffff5bb1);
        FF_SIMD(c4, d4, a4, b4, x_vec4[10], s13, 0xffff5bb1);

        FF_SIMD(b1, c1, d1, a1, x_vec1[11], s14, 0x895cd7be);
        FF_SIMD(b2, c2, d2, a2, x_vec2[11], s14, 0x895cd7be);
        FF_SIMD(b3, c3, d3, a3, x_vec3[11], s14, 0x895cd7be);
        FF_SIMD(b4, c4, d4, a4, x_vec4[11], s14, 0x895cd7be);

        FF_SIMD(a1, b1, c1, d1, x_vec1[12], s11, 0x6b901122);
        FF_SIMD(a2, b2, c2, d2, x_vec2[12], s11, 0x6b901122);
        FF_SIMD(a3, b3, c3, d3, x_vec3[12], s11, 0x6b901122);
        FF_SIMD(a4, b4, c4, d4, x_vec4[12], s11, 0x6b901122);

        FF_SIMD(d1, a1, b1, c1, x_vec1[13], s12, 0xfd987193);
        FF_SIMD(d2, a2, b2, c2, x_vec2[13], s12, 0xfd987193);
        FF_SIMD(d3, a3, b3, c3, x_vec3[13], s12, 0xfd987193);
        FF_SIMD(d4, a4, b4, c4, x_vec4[13], s12, 0xfd987193);

        FF_SIMD(c1, d1, a1, b1, x_vec1[14], s13, 0xa679438e);
        FF_SIMD(c2, d2, a2, b2, x_vec2[14], s13, 0xa679438e);
        FF_SIMD(c3, d3, a3, b3, x_vec3[14], s13, 0xa679438e);
        FF_SIMD(c4, d4, a4, b4, x_vec4[14], s13, 0xa679438e);

        FF_SIMD(b1, c1, d1, a1, x_vec1[15], s14, 0x49b40821);
        FF_SIMD(b2, c2, d2, a2, x_vec2[15], s14, 0x49b40821);
        FF_SIMD(b3, c3, d3, a3, x_vec3[15], s14, 0x49b40821);
        FF_SIMD(b4, c4, d4, a4, x_vec4[15], s14, 0x49b40821);

        // 第二轮 (Round 2) - 交织执行四组
        GG_SIMD(a1, b1, c1, d1, x_vec1[1], s21, 0xf61e2562);
        GG_SIMD(a2, b2, c2, d2, x_vec2[1], s21, 0xf61e2562);
        GG_SIMD(a3, b3, c3, d3, x_vec3[1], s21, 0xf61e2562);
        GG_SIMD(a4, b4, c4, d4, x_vec4[1], s21, 0xf61e2562);

        GG_SIMD(d1, a1, b1, c1, x_vec1[6], s22, 0xc040b340);
        GG_SIMD(d2, a2, b2, c2, x_vec2[6], s22, 0xc040b340);
        GG_SIMD(d3, a3, b3, c3, x_vec3[6], s22, 0xc040b340);
        GG_SIMD(d4, a4, b4, c4, x_vec4[6], s22, 0xc040b340);

        GG_SIMD(c1, d1, a1, b1, x_vec1[11], s23, 0x265e5a51);
        GG_SIMD(c2, d2, a2, b2, x_vec2[11], s23, 0x265e5a51);
        GG_SIMD(c3, d3, a3, b3, x_vec3[11], s23, 0x265e5a51);
        GG_SIMD(c4, d4, a4, b4, x_vec4[11], s23, 0x265e5a51);

        GG_SIMD(b1, c1, d1, a1, x_vec1[0], s24, 0xe9b6c7aa);
        GG_SIMD(b2, c2, d2, a2, x_vec2[0], s24, 0xe9b6c7aa);
        GG_SIMD(b3, c3, d3, a3, x_vec3[0], s24, 0xe9b6c7aa);
        GG_SIMD(b4, c4, d4, a4, x_vec4[0], s24, 0xe9b6c7aa);

        GG_SIMD(a1, b1, c1, d1, x_vec1[5], s21, 0xd62f105d);
        GG_SIMD(a2, b2, c2, d2, x_vec2[5], s21, 0xd62f105d);
        GG_SIMD(a3, b3, c3, d3, x_vec3[5], s21, 0xd62f105d);
        GG_SIMD(a4, b4, c4, d4, x_vec4[5], s21, 0xd62f105d);

        GG_SIMD(d1, a1, b1, c1, x_vec1[10], s22, 0x02441453);
        GG_SIMD(d2, a2, b2, c2, x_vec2[10], s22, 0x02441453);
        GG_SIMD(d3, a3, b3, c3, x_vec3[10], s22, 0x02441453);
        GG_SIMD(d4, a4, b4, c4, x_vec4[10], s22, 0x02441453);

        GG_SIMD(c1, d1, a1, b1, x_vec1[15], s23, 0xd8a1e681);
        GG_SIMD(c2, d2, a2, b2, x_vec2[15], s23, 0xd8a1e681);
        GG_SIMD(c3, d3, a3, b3, x_vec3[15], s23, 0xd8a1e681);
        GG_SIMD(c4, d4, a4, b4, x_vec4[15], s23, 0xd8a1e681);

        GG_SIMD(b1, c1, d1, a1, x_vec1[4], s24, 0xe7d3fbc8);
        GG_SIMD(b2, c2, d2, a2, x_vec2[4], s24, 0xe7d3fbc8);
        GG_SIMD(b3, c3, d3, a3, x_vec3[4], s24, 0xe7d3fbc8);
        GG_SIMD(b4, c4, d4, a4, x_vec4[4], s24, 0xe7d3fbc8);

        GG_SIMD(a1, b1, c1, d1, x_vec1[9], s21, 0x21e1cde6);
        GG_SIMD(a2, b2, c2, d2, x_vec2[9], s21, 0x21e1cde6);
        GG_SIMD(a3, b3, c3, d3, x_vec3[9], s21, 0x21e1cde6);
        GG_SIMD(a4, b4, c4, d4, x_vec4[9], s21, 0x21e1cde6);

        GG_SIMD(d1, a1, b1, c1, x_vec1[14], s22, 0xc33707d6);
        GG_SIMD(d2, a2, b2, c2, x_vec2[14], s22, 0xc33707d6);
        GG_SIMD(d3, a3, b3, c3, x_vec3[14], s22, 0xc33707d6);
        GG_SIMD(d4, a4, b4, c4, x_vec4[14], s22, 0xc33707d6);

        GG_SIMD(c1, d1, a1, b1, x_vec1[3], s23, 0xf4d50d87);
        GG_SIMD(c2, d2, a2, b2, x_vec2[3], s23, 0xf4d50d87);
        GG_SIMD(c3, d3, a3, b3, x_vec3[3], s23, 0xf4d50d87);
        GG_SIMD(c4, d4, a4, b4, x_vec4[3], s23, 0xf4d50d87);

        GG_SIMD(b1, c1, d1, a1, x_vec1[8], s24, 0x455a14ed);
        GG_SIMD(b2, c2, d2, a2, x_vec2[8], s24, 0x455a14ed);
        GG_SIMD(b3, c3, d3, a3, x_vec3[8], s24, 0x455a14ed);
        GG_SIMD(b4, c4, d4, a4, x_vec4[8], s24, 0x455a14ed);

        GG_SIMD(a1, b1, c1, d1, x_vec1[13], s21, 0xa9e3e905);
        GG_SIMD(a2, b2, c2, d2, x_vec2[13], s21, 0xa9e3e905);
        GG_SIMD(a3, b3, c3, d3, x_vec3[13], s21, 0xa9e3e905);
        GG_SIMD(a4, b4, c4, d4, x_vec4[13], s21, 0xa9e3e905);

        GG_SIMD(d1, a1, b1, c1, x_vec1[2], s22, 0xfcefa3f8);
        GG_SIMD(d2, a2, b2, c2, x_vec2[2], s22, 0xfcefa3f8);
        GG_SIMD(d3, a3, b3, c3, x_vec3[2], s22, 0xfcefa3f8);
        GG_SIMD(d4, a4, b4, c4, x_vec4[2], s22, 0xfcefa3f8);

        GG_SIMD(c1, d1, a1, b1, x_vec1[7], s23, 0x676f02d9);
        GG_SIMD(c2, d2, a2, b2, x_vec2[7], s23, 0x676f02d9);
        GG_SIMD(c3, d3, a3, b3, x_vec3[7], s23, 0x676f02d9);
        GG_SIMD(c4, d4, a4, b4, x_vec4[7], s23, 0x676f02d9);

        GG_SIMD(b1, c1, d1, a1, x_vec1[12], s24, 0x8d2a4c8a);
        GG_SIMD(b2, c2, d2, a2, x_vec2[12], s24, 0x8d2a4c8a);
        GG_SIMD(b3, c3, d3, a3, x_vec3[12], s24, 0x8d2a4c8a);
        GG_SIMD(b4, c4, d4, a4, x_vec4[12], s24, 0x8d2a4c8a);

        // 第三轮 (Round 3) - 交织执行四组
        HH_SIMD(a1, b1, c1, d1, x_vec1[5], s31, 0xfffa3942);
        HH_SIMD(a2, b2, c2, d2, x_vec2[5], s31, 0xfffa3942);
        HH_SIMD(a3, b3, c3, d3, x_vec3[5], s31, 0xfffa3942);
        HH_SIMD(a4, b4, c4, d4, x_vec4[5], s31, 0xfffa3942);

        HH_SIMD(d1, a1, b1, c1, x_vec1[8], s32, 0x8771f681);
        HH_SIMD(d2, a2, b2, c2, x_vec2[8], s32, 0x8771f681);
        HH_SIMD(d3, a3, b3, c3, x_vec3[8], s32, 0x8771f681);
        HH_SIMD(d4, a4, b4, c4, x_vec4[8], s32, 0x8771f681);

        HH_SIMD(c1, d1, a1, b1, x_vec1[11], s33, 0x6d9d6122);
        HH_SIMD(c2, d2, a2, b2, x_vec2[11], s33, 0x6d9d6122);
        HH_SIMD(c3, d3, a3, b3, x_vec3[11], s33, 0x6d9d6122);
        HH_SIMD(c4, d4, a4, b4, x_vec4[11], s33, 0x6d9d6122);

        HH_SIMD(b1, c1, d1, a1, x_vec1[14], s34, 0xfde5380c);
        HH_SIMD(b2, c2, d2, a2, x_vec2[14], s34, 0xfde5380c);
        HH_SIMD(b3, c3, d3, a3, x_vec3[14], s34, 0xfde5380c);
        HH_SIMD(b4, c4, d4, a4, x_vec4[14], s34, 0xfde5380c);

        HH_SIMD(a1, b1, c1, d1, x_vec1[1], s31, 0xa4beea44);
        HH_SIMD(a2, b2, c2, d2, x_vec2[1], s31, 0xa4beea44);
        HH_SIMD(a3, b3, c3, d3, x_vec3[1], s31, 0xa4beea44);
        HH_SIMD(a4, b4, c4, d4, x_vec4[1], s31, 0xa4beea44);

        HH_SIMD(d1, a1, b1, c1, x_vec1[4], s32, 0x4bdecfa9);
        HH_SIMD(d2, a2, b2, c2, x_vec2[4], s32, 0x4bdecfa9);
        HH_SIMD(d3, a3, b3, c3, x_vec3[4], s32, 0x4bdecfa9);
        HH_SIMD(d4, a4, b4, c4, x_vec4[4], s32, 0x4bdecfa9);

        HH_SIMD(c1, d1, a1, b1, x_vec1[7], s33, 0xf6bb4b60);
        HH_SIMD(c2, d2, a2, b2, x_vec2[7], s33, 0xf6bb4b60);
        HH_SIMD(c3, d3, a3, b3, x_vec3[7], s33, 0xf6bb4b60);
        HH_SIMD(c4, d4, a4, b4, x_vec4[7], s33, 0xf6bb4b60);

        HH_SIMD(b1, c1, d1, a1, x_vec1[10], s34, 0xbebfbc70);
        HH_SIMD(b2, c2, d2, a2, x_vec2[10], s34, 0xbebfbc70);
        HH_SIMD(b3, c3, d3, a3, x_vec3[10], s34, 0xbebfbc70);
        HH_SIMD(b4, c4, d4, a4, x_vec4[10], s34, 0xbebfbc70);

        HH_SIMD(a1, b1, c1, d1, x_vec1[13], s31, 0x289b7ec6);
        HH_SIMD(a2, b2, c2, d2, x_vec2[13], s31, 0x289b7ec6);
        HH_SIMD(a3, b3, c3, d3, x_vec3[13], s31, 0x289b7ec6);
        HH_SIMD(a4, b4, c4, d4, x_vec4[13], s31, 0x289b7ec6);

        HH_SIMD(d1, a1, b1, c1, x_vec1[0], s32, 0xeaa127fa);
        HH_SIMD(d2, a2, b2, c2, x_vec2[0], s32, 0xeaa127fa);
        HH_SIMD(d3, a3, b3, c3, x_vec3[0], s32, 0xeaa127fa);
        HH_SIMD(d4, a4, b4, c4, x_vec4[0], s32, 0xeaa127fa);

        HH_SIMD(c1, d1, a1, b1, x_vec1[3], s33, 0xd4ef3085);
        HH_SIMD(c2, d2, a2, b2, x_vec2[3], s33, 0xd4ef3085);
        HH_SIMD(c3, d3, a3, b3, x_vec3[3], s33, 0xd4ef3085);
        HH_SIMD(c4, d4, a4, b4, x_vec4[3], s33, 0xd4ef3085);

        HH_SIMD(b1, c1, d1, a1, x_vec1[6], s34, 0x04881d05);
        HH_SIMD(b2, c2, d2, a2, x_vec2[6], s34, 0x04881d05);
        HH_SIMD(b3, c3, d3, a3, x_vec3[6], s34, 0x04881d05);
        HH_SIMD(b4, c4, d4, a4, x_vec4[6], s34, 0x04881d05);

        HH_SIMD(a1, b1, c1, d1, x_vec1[9], s31, 0xd9d4d039);
        HH_SIMD(a2, b2, c2, d2, x_vec2[9], s31, 0xd9d4d039);
        HH_SIMD(a3, b3, c3, d3, x_vec3[9], s31, 0xd9d4d039);
        HH_SIMD(a4, b4, c4, d4, x_vec4[9], s31, 0xd9d4d039);

        HH_SIMD(d1, a1, b1, c1, x_vec1[12], s32, 0xe6db99e5);
        HH_SIMD(d2, a2, b2, c2, x_vec2[12], s32, 0xe6db99e5);
        HH_SIMD(d3, a3, b3, c3, x_vec3[12], s32, 0xe6db99e5);
        HH_SIMD(d4, a4, b4, c4, x_vec4[12], s32, 0xe6db99e5);

        HH_SIMD(c1, d1, a1, b1, x_vec1[15], s33, 0x1fa27cf8);
        HH_SIMD(c2, d2, a2, b2, x_vec2[15], s33, 0x1fa27cf8);
        HH_SIMD(c3, d3, a3, b3, x_vec3[15], s33, 0x1fa27cf8);
        HH_SIMD(c4, d4, a4, b4, x_vec4[15], s33, 0x1fa27cf8);

        HH_SIMD(b1, c1, d1, a1, x_vec1[2], s34, 0xc4ac5665);
        HH_SIMD(b2, c2, d2, a2, x_vec2[2], s34, 0xc4ac5665);
        HH_SIMD(b3, c3, d3, a3, x_vec3[2], s34, 0xc4ac5665);
        HH_SIMD(b4, c4, d4, a4, x_vec4[2], s34, 0xc4ac5665);

        // 第四轮 (Round 4) - 交织执行四组
        II_SIMD(a1, b1, c1, d1, x_vec1[0], s41, 0xf4292244);
        II_SIMD(a2, b2, c2, d2, x_vec2[0], s41, 0xf4292244);
        II_SIMD(a3, b3, c3, d3, x_vec3[0], s41, 0xf4292244);
        II_SIMD(a4, b4, c4, d4, x_vec4[0], s41, 0xf4292244);

        II_SIMD(d1, a1, b1, c1, x_vec1[7], s42, 0x432aff97);
        II_SIMD(d2, a2, b2, c2, x_vec2[7], s42, 0x432aff97);
        II_SIMD(d3, a3, b3, c3, x_vec3[7], s42, 0x432aff97);
        II_SIMD(d4, a4, b4, c4, x_vec4[7], s42, 0x432aff97);

        II_SIMD(c1, d1, a1, b1, x_vec1[14], s43, 0xab9423a7);
        II_SIMD(c2, d2, a2, b2, x_vec2[14], s43, 0xab9423a7);
        II_SIMD(c3, d3, a3, b3, x_vec3[14], s43, 0xab9423a7);
        II_SIMD(c4, d4, a4, b4, x_vec4[14], s43, 0xab9423a7);

        II_SIMD(b1, c1, d1, a1, x_vec1[5], s44, 0xfc93a039);
        II_SIMD(b2, c2, d2, a2, x_vec2[5], s44, 0xfc93a039);
        II_SIMD(b3, c3, d3, a3, x_vec3[5], s44, 0xfc93a039);
        II_SIMD(b4, c4, d4, a4, x_vec4[5], s44, 0xfc93a039);

        II_SIMD(a1, b1, c1, d1, x_vec1[12], s41, 0x655b59c3);
        II_SIMD(a2, b2, c2, d2, x_vec2[12], s41, 0x655b59c3);
        II_SIMD(a3, b3, c3, d3, x_vec3[12], s41, 0x655b59c3);
        II_SIMD(a4, b4, c4, d4, x_vec4[12], s41, 0x655b59c3);

        II_SIMD(d1, a1, b1, c1, x_vec1[3], s42, 0x8f0ccc92);
        II_SIMD(d2, a2, b2, c2, x_vec2[3], s42, 0x8f0ccc92);
        II_SIMD(d3, a3, b3, c3, x_vec3[3], s42, 0x8f0ccc92);
        II_SIMD(d4, a4, b4, c4, x_vec4[3], s42, 0x8f0ccc92);

        II_SIMD(c1, d1, a1, b1, x_vec1[10], s43, 0xffeff47d);
        II_SIMD(c2, d2, a2, b2, x_vec2[10], s43, 0xffeff47d);
        II_SIMD(c3, d3, a3, b3, x_vec3[10], s43, 0xffeff47d);
        II_SIMD(c4, d4, a4, b4, x_vec4[10], s43, 0xffeff47d);

        II_SIMD(b1, c1, d1, a1, x_vec1[1], s44, 0x85845dd1);
        II_SIMD(b2, c2, d2, a2, x_vec2[1], s44, 0x85845dd1);
        II_SIMD(b3, c3, d3, a3, x_vec3[1], s44, 0x85845dd1);
        II_SIMD(b4, c4, d4, a4, x_vec4[1], s44, 0x85845dd1);

        II_SIMD(a1, b1, c1, d1, x_vec1[8], s41, 0x6fa87e4f);
        II_SIMD(a2, b2, c2, d2, x_vec2[8], s41, 0x6fa87e4f);
        II_SIMD(a3, b3, c3, d3, x_vec3[8], s41, 0x6fa87e4f);
        II_SIMD(a4, b4, c4, d4, x_vec4[8], s41, 0x6fa87e4f);

        II_SIMD(d1, a1, b1, c1, x_vec1[15], s42, 0xfe2ce6e0);
        II_SIMD(d2, a2, b2, c2, x_vec2[15], s42, 0xfe2ce6e0);
        II_SIMD(d3, a3, b3, c3, x_vec3[15], s42, 0xfe2ce6e0);
        II_SIMD(d4, a4, b4, c4, x_vec4[15], s42, 0xfe2ce6e0);

        II_SIMD(c1, d1, a1, b1, x_vec1[6], s43, 0xa3014314);
        II_SIMD(c2, d2, a2, b2, x_vec2[6], s43, 0xa3014314);
        II_SIMD(c3, d3, a3, b3, x_vec3[6], s43, 0xa3014314);
        II_SIMD(c4, d4, a4, b4, x_vec4[6], s43, 0xa3014314);

        II_SIMD(b1, c1, d1, a1, x_vec1[13], s44, 0x4e0811a1);
        II_SIMD(b2, c2, d2, a2, x_vec2[13], s44, 0x4e0811a1);
        II_SIMD(b3, c3, d3, a3, x_vec3[13], s44, 0x4e0811a1);
        II_SIMD(b4, c4, d4, a4, x_vec4[13], s44, 0x4e0811a1);

        II_SIMD(a1, b1, c1, d1, x_vec1[4], s41, 0xf7537e82);
        II_SIMD(a2, b2, c2, d2, x_vec2[4], s41, 0xf7537e82);
        II_SIMD(a3, b3, c3, d3, x_vec3[4], s41, 0xf7537e82);
        II_SIMD(a4, b4, c4, d4, x_vec4[4], s41, 0xf7537e82);

        II_SIMD(d1, a1, b1, c1, x_vec1[11], s42, 0xbd3af235);
        II_SIMD(d2, a2, b2, c2, x_vec2[11], s42, 0xbd3af235);
        II_SIMD(d3, a3, b3, c3, x_vec3[11], s42, 0xbd3af235);
        II_SIMD(d4, a4, b4, c4, x_vec4[11], s42, 0xbd3af235);

        II_SIMD(c1, d1, a1, b1, x_vec1[2], s43, 0x2ad7d2bb);
        II_SIMD(c2, d2, a2, b2, x_vec2[2], s43, 0x2ad7d2bb);
        II_SIMD(c3, d3, a3, b3, x_vec3[2], s43, 0x2ad7d2bb);
        II_SIMD(c4, d4, a4, b4, x_vec4[2], s43, 0x2ad7d2bb);

        II_SIMD(b1, c1, d1, a1, x_vec1[9], s44, 0xeb86d391);
        II_SIMD(b2, c2, d2, a2, x_vec2[9], s44, 0xeb86d391);
        II_SIMD(b3, c3, d3, a3, x_vec3[9], s44, 0xeb86d391);
        II_SIMD(b4, c4, d4, a4, x_vec4[9], s44, 0xeb86d391);

        // 更新状态
        a1 = vaddq_u32(a1, aa1);
        b1 = vaddq_u32(b1, bb1);
        c1 = vaddq_u32(c1, cc1);
        d1 = vaddq_u32(d1, dd1);
        
        a2 = vaddq_u32(a2, aa2);
        b2 = vaddq_u32(b2, bb2);
        c2 = vaddq_u32(c2, cc2);
        d2 = vaddq_u32(d2, dd2);
        
        a3 = vaddq_u32(a3, aa3);
        b3 = vaddq_u32(b3, bb3);
        c3 = vaddq_u32(c3, cc3);
        d3 = vaddq_u32(d3, dd3);
        
        a4 = vaddq_u32(a4, aa4);
        b4 = vaddq_u32(b4, bb4);
        c4 = vaddq_u32(c4, cc4);
        d4 = vaddq_u32(d4, dd4);

        // 更新保存的初始值，为下一个块做准备
        aa1 = a1; bb1 = b1; cc1 = c1; dd1 = d1;
        aa2 = a2; bb2 = b2; cc2 = c2; dd2 = d2;
        aa3 = a3; bb3 = b3; cc3 = c3; dd3 = d3;
        aa4 = a4; bb4 = b4; cc4 = c4; dd4 = d4;
    }

    // 存储结果
    uint32_t result_a1[4], result_b1[4], result_c1[4], result_d1[4];
    uint32_t result_a2[4], result_b2[4], result_c2[4], result_d2[4];
    uint32_t result_a3[4], result_b3[4], result_c3[4], result_d3[4];
    uint32_t result_a4[4], result_b4[4], result_c4[4], result_d4[4];
    
    vst1q_u32(result_a1, a1);
    vst1q_u32(result_b1, b1);
    vst1q_u32(result_c1, c1);
    vst1q_u32(result_d1, d1);
    
    vst1q_u32(result_a2, a2);
    vst1q_u32(result_b2, b2);
    vst1q_u32(result_c2, c2);
    vst1q_u32(result_d2, d2);
    
    vst1q_u32(result_a3, a3);
    vst1q_u32(result_b3, b3);
    vst1q_u32(result_c3, c3);
    vst1q_u32(result_d3, d3);
    
    vst1q_u32(result_a4, a4);
    vst1q_u32(result_b4, b4);
    vst1q_u32(result_c4, c4);
    vst1q_u32(result_d4, d4);
    
    // 字节序转换并复制到结果数组
    for (int i = 0; i < 4; i++) {
        // 第一组的结果
        states[i][0] = ((result_a1[i] & 0xFF) << 24) |
                      ((result_a1[i] & 0xFF00) << 8) |
                      ((result_a1[i] & 0xFF0000) >> 8) |
                      ((result_a1[i] & 0xFF000000) >> 24);
        states[i][1] = ((result_b1[i] & 0xFF) << 24) |
                      ((result_b1[i] & 0xFF00) << 8) |
                      ((result_b1[i] & 0xFF0000) >> 8) |
                      ((result_b1[i] & 0xFF000000) >> 24);
        states[i][2] = ((result_c1[i] & 0xFF) << 24) |
                      ((result_c1[i] & 0xFF00) << 8) |
                      ((result_c1[i] & 0xFF0000) >> 8) |
                      ((result_c1[i] & 0xFF000000) >> 24);
        states[i][3] = ((result_d1[i] & 0xFF) << 24) |
                      ((result_d1[i] & 0xFF00) << 8) |
                      ((result_d1[i] & 0xFF0000) >> 8) |
                      ((result_d1[i] & 0xFF000000) >> 24);
        
        // 第二组的结果
        states[i+4][0] = ((result_a2[i] & 0xFF) << 24) |
                        ((result_a2[i] & 0xFF00) << 8) |
                        ((result_a2[i] & 0xFF0000) >> 8) |
                        ((result_a2[i] & 0xFF000000) >> 24);
        states[i+4][1] = ((result_b2[i] & 0xFF) << 24) |
                        ((result_b2[i] & 0xFF00) << 8) |
                        ((result_b2[i] & 0xFF0000) >> 8) |
                        ((result_b2[i] & 0xFF000000) >> 24);
        states[i+4][2] = ((result_c2[i] & 0xFF) << 24) |
                        ((result_c2[i] & 0xFF00) << 8) |
                        ((result_c2[i] & 0xFF0000) >> 8) |
                        ((result_c2[i] & 0xFF000000) >> 24);
        states[i+4][3] = ((result_d2[i] & 0xFF) << 24) |
                        ((result_d2[i] & 0xFF00) << 8) |
                        ((result_d2[i] & 0xFF0000) >> 8) |
                        ((result_d2[i] & 0xFF000000) >> 24);
                        
        // 第三组的结果
        states[i+8][0] = ((result_a3[i] & 0xFF) << 24) |
                        ((result_a3[i] & 0xFF00) << 8) |
                        ((result_a3[i] & 0xFF0000) >> 8) |
                        ((result_a3[i] & 0xFF000000) >> 24);
        states[i+8][1] = ((result_b3[i] & 0xFF) << 24) |
                        ((result_b3[i] & 0xFF00) << 8) |
                        ((result_b3[i] & 0xFF0000) >> 8) |
                        ((result_b3[i] & 0xFF000000) >> 24);
        states[i+8][2] = ((result_c3[i] & 0xFF) << 24) |
                        ((result_c3[i] & 0xFF00) << 8) |
                        ((result_c3[i] & 0xFF0000) >> 8) |
                        ((result_c3[i] & 0xFF000000) >> 24);
        states[i+8][3] = ((result_d3[i] & 0xFF) << 24) |
                        ((result_d3[i] & 0xFF00) << 8) |
                        ((result_d3[i] & 0xFF0000) >> 8) |
                        ((result_d3[i] & 0xFF000000) >> 24);
        
        // 第四组的结果
        states[i+12][0] = ((result_a4[i] & 0xFF) << 24) |
                         ((result_a4[i] & 0xFF00) << 8) |
                         ((result_a4[i] & 0xFF0000) >> 8) |
                         ((result_a4[i] & 0xFF000000) >> 24);
        states[i+12][1] = ((result_b4[i] & 0xFF) << 24) |
                         ((result_b4[i] & 0xFF00) << 8) |
                         ((result_b4[i] & 0xFF0000) >> 8) |
                         ((result_b4[i] & 0xFF000000) >> 24);
        states[i+12][2] = ((result_c4[i] & 0xFF) << 24) |
                         ((result_c4[i] & 0xFF00) << 8) |
                         ((result_c4[i] & 0xFF0000) >> 8) |
                         ((result_c4[i] & 0xFF000000) >> 24);
        states[i+12][3] = ((result_d4[i] & 0xFF) << 24) |
                         ((result_d4[i] & 0xFF00) << 8) |
                         ((result_d4[i] & 0xFF0000) >> 8) |
                         ((result_d4[i] & 0xFF000000) >> 24);
    }
    
    // 释放内存
    for (int i = 0; i < 16; i++) {
        delete[] paddedMessages[i];
    }
}