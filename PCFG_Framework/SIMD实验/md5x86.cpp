#include "md5x86.h"
#include <iomanip>
#include <assert.h>
#include <chrono>
#include<algorithm>
using namespace std;
using namespace chrono;

/**
 * StringProcess: �����������ַ���ת����MD5�����������Ϣ����
 * @param input ����
 * @param[out] n_byte ���ڸ������ߴ��ݶ���ķ���ֵ��������Byte����ĳ���
 * @return Byte��Ϣ����
 */
Byte* StringProcess(string input, int* n_byte)
{
	// ��������ַ���ת��ΪByteΪ��λ������
	Byte* blocks = (Byte*)input.c_str();
	int length = input.length();

	// ����ԭʼ��Ϣ���ȣ���bΪ��λ��
	int bitLength = length * 8;

	// paddingBits: ԭʼ��Ϣ��Ҫ��padding���ȣ���bitΪ��λ��
	// ���ڸ�������Ϣ�����䲹����length%512==448Ϊֹ
	// ��Ҫע����ǣ������������Ϣ����length%512==448��Ҳ��Ҫ��pad 512bits
	int paddingBits = bitLength % 512;
	if (paddingBits > 448)
	{
		paddingBits = 512 - (paddingBits - 448);
	}
	else if (paddingBits < 448)
	{
		paddingBits = 448 - paddingBits;
	}
	else if (paddingBits == 448)
	{
		paddingBits = 512;
	}

	// ԭʼ��Ϣ��Ҫ��padding���ȣ���ByteΪ��λ��
	int paddingBytes = paddingBits / 8;
	// �������յ��ֽ�����
	// length + paddingBytes + 8:
	// 1. lengthΪԭʼ��Ϣ�ĳ��ȣ�bits��
	// 2. paddingBytesΪԭʼ��Ϣ��Ҫ��padding���ȣ�Bytes��
	// 3. ��pad��length%512==448֮����Ҫ���⸽��64bits��ԭʼ��Ϣ���ȣ���8��bytes
	int paddedLength = length + paddingBytes + 8;
	Byte* paddedMessage = new Byte[paddedLength];

	// ����ԭʼ��Ϣ
	memcpy(paddedMessage, blocks, length);

	// ��������ֽڡ����ʱ����һλΪ1�����������λ��Ϊ0��
	// ���Ե�һ��byte��0x80
	paddedMessage[length] = 0x80;							 // ����һ��0x80�ֽ�
	memset(paddedMessage + length + 1, 0, paddingBytes - 1); // ���0�ֽ�

	// ������Ϣ���ȣ�64���أ�С�˸�ʽ��
	for (int i = 0; i < 8; ++i)
	{
		// �ر�ע��˴�Ӧ����bitLengthת��Ϊuint64_t
		// �����length��ԭʼ��Ϣ�ĳ���
		paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
	}

	// ��֤�����Ƿ�����Ҫ�󡣴�ʱ����Ӧ����512bit�ı���
	//int residual = 8 * paddedLength % 512;
	// assert(residual == 0);

	// �����+���ӳ���֮����Ϣ����Ϊn_blocks��512bit�Ĳ���
	*n_byte = paddedLength;
	return paddedMessage;
}




/**
 * MD5Hash: �����������ַ���ת����MD5
 * @param input ����
 * @param[out] state ���ڸ������ߴ��ݶ���ķ���ֵ�������յĻ�������Ҳ����MD5�Ľ��
 * @return Byte��Ϣ����
 */
void MD5Hash(string input, bit32* state)
{

	Byte* paddedMessage;
	int* messageLength = new int[1];///B
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

	// ��block�ظ���state
	for (int i = 0; i < n_blocks; i += 1)
	{
		bit32 x[16];

		// ����Ĵ������������Ͻ�Ϊ����
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
		GG(d, a, b, c, x[10], s22, 0x2441453);
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
		HH(b, c, d, a, x[6], s34, 0x4881d05);
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

	// ����Ĵ������������Ͻ�Ϊ����
	for (int i = 0; i < 4; i++)
	{
		uint32_t value = state[i];
		state[i] = ((value & 0xff) << 24) |		 // ������ֽ��Ƶ����λ
			((value & 0xff00) << 8) |	 // ���ε��ֽ�����
			((value & 0xff0000) >> 8) |	 // ���θ��ֽ�����
			((value & 0xff000000) >> 24); // ������ֽ��Ƶ����λ
	}

	// ������յ�hash���
	// for (int i1 = 0; i1 < 4; i1 += 1)
	// {
	// 	cout << std::setw(8) << std::setfill('0') << hex << state[i1];
	// }
	// cout << endl;

	// �ͷŶ�̬������ڴ�
	// ʵ��SIMD�����㷨��ʱ��Ҳ��ǵü�ʱ�����ڴ棡
	delete[] paddedMessage;
	delete[] messageLength;
	//-		states	0x000000f745eff668 {0x00000226ea6e7060 {0x23abd951}}	unsigned int * *

}

void MD5Hash_SIMD8(const string inputs[8], bit32 states[8][4]) {
	// Ϊ8������׼��������
	Byte* paddedMessages[8];
	int paddedLengths[8];
	int n_blocks[8];

	// ����ÿ���ַ�������ʼ��״̬
	for (int i = 0; i < 8; i++) {
		paddedMessages[i] = StringProcess(inputs[i], &paddedLengths[i]);
		n_blocks[i] = paddedLengths[i] / 64;
		states[i][0] = 0x67452301;
		states[i][1] = 0xefcdab89;
		states[i][2] = 0x98badcfe;
		states[i][3] = 0x10325476;
	}

	// ��ȡ������
	int max_blocks = *max_element(n_blocks, n_blocks + 8);

	// ����SIMD��ʼ״̬
	__m256i a = _mm256_set1_epi32(0x67452301);
	__m256i b = _mm256_set1_epi32(0xefcdab89);
	__m256i c = _mm256_set1_epi32(0x98badcfe);
	__m256i d = _mm256_set1_epi32(0x10325476);
	__m256i aa = a, bb = b, cc = c, dd = d;

	// ��ʱ����������ת������
	bit32 x[8];
	__m256i x_vec[16];

	// ����ÿ����
	for (int i = 0; i < max_blocks; ++i) {
		// Ϊÿ����Ϣ׼��16��32λ��
		for (int j = 0; j < 16; j++) {
			int msg_offset = i * 64 + j * 4;

			// ��ÿ�������������
			for (int k = 0; k < 8; k++) {
				// �����Ѿ�����������룬�ظ�ʹ�����һ���������
				if (i < n_blocks[k]) {
					x[k] = (paddedMessages[k][msg_offset]) |
						(paddedMessages[k][msg_offset + 1] << 8) |
						(paddedMessages[k][msg_offset + 2] << 16) |
						(paddedMessages[k][msg_offset + 3] << 24);
				}
				else {
					// �����ǰ�����Ѵ����꣬ʹ��0���
					x[k] = 0;
				}
			}

			// ��8��32λ�������ص�һ��256λSIMD�Ĵ�����
			x_vec[j] = _mm256_loadu_si256((__m256i*)x);
		}

		// ��һ��
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

		// �ڶ���
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

		// ������
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

		// ������
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

		// �����м��ϣֵ
		a = _mm256_add_epi32(a, aa);
		b = _mm256_add_epi32(b, bb);
		c = _mm256_add_epi32(c, cc);
		d = _mm256_add_epi32(d, dd);
	}

	// �洢�����ת��Ϊ��ȷ���ֽ�˳��
	bit32 result_a[8], result_b[8], result_c[8], result_d[8];
	_mm256_storeu_si256((__m256i*)result_a, a);
	_mm256_storeu_si256((__m256i*)result_b, b);
	_mm256_storeu_si256((__m256i*)result_c, c);
	_mm256_storeu_si256((__m256i*)result_d, d);

	// �����ֽ�˳��ת�������Ƶ��������
	for (int i = 0; i < 8; i++) {
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

	// �ͷ��ڴ�
	for (int i = 0; i < 8; i++) {
		delete[] paddedMessages[i];
	}
}
