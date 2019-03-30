/*
the reference: https://en.wikipedia.org/wiki/MD5
*/

#include "sha.h"


#define BLOCK 64
#define F(x,y,z) (((x) & (y)) | (~(x) & (z)))
#define G(x,y,z) (((x) & (z)) | (~(z) & (y)))
#define H(x,y,z) ((x) ^ (y) ^ (z))
#define I(x,y,z) ((y) ^ ((x) | ~(z)))
#define rotate_left(a,b) (((a) << (b)) | ((a) >> (32-(b))))


__constant__ unsigned int R[64] = {
    7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
    5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
    4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
    6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21
};

__constant__ unsigned int INIT[4] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476 };

__constant__ unsigned int L[16] = { 0 };
__constant__ unsigned int K[64] = { 
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};


__global__ void md5(unsigned int* d) {
    unsigned int h[4];
    h[0] = 0x67452301;
    h[1] = 0xEFCDAB89;
    h[2] = 0x98BADCFE;
    h[3] = 0x10325476;

    unsigned int i = 0;
    unsigned int f = 0, g = 0;
    for (; i < 16; ++i) {
        f = F(h[1], h[2], h[3]);
        g = i;

        f += f + h[0] + K[i] + L[g];
        h[0] = h[3];
        h[3] = h[2];
        h[2] = h[1];
        h[1] += rotate_left(f, R[i]);
    }

    for (; i < 32; ++i) {
        f = G(h[1], h[2], h[3]);
        g = (5 * i + 1) % 16;

        f += f + h[0] + K[i] + L[g];
        h[0] = h[3];
        h[3] = h[2];
        h[2] = h[1];
        h[1] += rotate_left(f, R[i]);
    }

    for (; i < 48; ++i) {
        f = H(h[1], h[2], h[3]);
        g = (3 * i + 5) % 16;;

        f += f + h[0] + K[i] + L[g];
        h[0] = h[3];
        h[3] = h[2];
        h[2] = h[1];
        h[1] += rotate_left(f, R[i]);
    }

    for (; i < 64; ++i) {
        f = I(h[1], h[2], h[3]);
        g = (7 * i) % 16;;

        f += f + h[0] + K[i] + L[g];
        h[0] = h[3];
        h[3] = h[2];
        h[2] = h[1];
        h[1] += rotate_left(f, R[i]);
    }

#pragma unroll
    for (unsigned int i = 0; i < 4; ++i) {
        d[i] = INIT[i] + h[i];
    }
}


void paddle_bits_md5(const std::string &_input) {
    std::string c = "";
    static unsigned int N[16];

    for (unsigned int i = 0; i < _input.length(); ++i) {
        c += std::bitset<8>(_input[i]).to_string();
    }

    c += "10000000"; // 0x80
    int len = c.length();
    c += std::string(448 - len, '0');
    std::string j = std::bitset<8>(_input.length() * 8).to_string();
    c += std::string(64 - j.length(), '0');
    c += j;

    for (unsigned int i = 0; i < 16; ++i) {
        N[i] = std::stoll(c.substr(32 * i, 32), nullptr, 2);
    }
    cudaMemcpyToSymbol(L, N, sizeof(unsigned int) * 16); // update device constant variable
}

extern "C" const char* MD5(const char *input) {
    cudaDeviceReset(); // clear all existing allcations on device in case exception happens.
    
    paddle_bits_md5(std::string(input));
    dim3 block_size(1, 1);
    dim3 grid_size(1);

    unsigned int *d_s;
    cudaMalloc(&d_s, sizeof(unsigned int) * 4);

    md5 << <1, 1 >> > (d_s);
    cudaDeviceSynchronize();

    unsigned int h_s[4] = { 0 };
    cudaMemcpy(h_s, d_s, sizeof(unsigned int) * 4, cudaMemcpyDeviceToHost);
    cudaFree(d_s);

    std::stringstream stream;
    for (unsigned int i = 0; i < 4; ++i) {
        stream << std::hex << std::setfill('0') << std::setw(8) << h_s[i];
    }
    char *ch = new char[stream.str().length() + 1];
    strcpy(ch, stream.str().c_str());
    return ch;
}