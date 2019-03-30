/*
the reference: https://en.wikipedia.org/wiki/MD5
*/

#include "sha.h"


#define BLOCK 64
#define rotate_right(a,b) (((a) >> (b)) | ((a) << (32-(b))))
#define ch(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define maj(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define sigma_0(x) (rotate_right(x,2) ^ rotate_right(x,13) ^ rotate_right(x,22))
#define sigma_1(x) (rotate_right(x,6) ^ rotate_right(x,11) ^ rotate_right(x,25))
#define ep_0(x) (rotate_right(x,7) ^ rotate_right(x,18) ^ ((x) >> 3))
#define ep_1(x) (rotate_right(x,17) ^ rotate_right(x,19) ^ ((x) >> 10))


__constant__ unsigned int H[8] = { 
    0x6a09e667,
    0xbb67ae85,
    0x3c6ef372,
    0xa54ff53a,
    0x510e527f,
    0x9b05688c,
    0x1f83d9ab,
    0x5be0cd19
};


__constant__ unsigned int K[BLOCK] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__constant__ unsigned int L[16] = { 0 };


__global__ void sha256(unsigned int* d ) {
    __shared__ unsigned int temp[BLOCK];
    unsigned int h[8];
#pragma unroll
    for (unsigned int i = 0; i < 16; ++i) temp[i] = L[i];
#pragma unroll
    for (unsigned int i = 16; i < BLOCK; ++i) {
        temp[i] = ep_1(temp[i - 2]) + temp[i - 7] + ep_0(temp[i - 15]) + temp[i - 16];
    }
    __syncthreads();

        
// #pragma unroll
    //for (unsigned int i = 0; i < 8; ++i) h[i] = H[i];
    reinterpret_cast<uint4*>(h)[0] = reinterpret_cast<uint4*>(H)[0];
    reinterpret_cast<uint4*>(h)[1] = reinterpret_cast<uint4*>(H)[1];
#pragma unroll
    for (unsigned int i = 0; i < BLOCK; ++i) {
        unsigned int z_d = h[7] + temp[i] + K[i] + ch(h[4], h[5], h[6]) + sigma_1(h[4]);
        unsigned int z_a = maj(h[0], h[1], h[2]) + sigma_0(h[0]);

        h[7] = h[6];
        h[6] = h[5];
        h[5] = h[4];
        h[4] = h[3] + z_d;
        h[3] = h[2];
        h[2] = h[1];
        h[1] = h[0];
        h[0] = z_a + z_d;
    }

#pragma unroll
    for (unsigned int i = 0; i < 8; ++i) {
        h[i] += H[i];
    }

    reinterpret_cast<uint4*>(d)[0] = reinterpret_cast<uint4*>(h)[0];
    reinterpret_cast<uint4*>(d)[1] = reinterpret_cast<uint4*>(h)[1];
}


void paddle_bits_256(const std::string &_input) {
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


extern "C" const char* SHA256(const char *input) {
    cudaDeviceReset(); // clear all existing allcations on device in case exception happens.
    
    paddle_bits_256(std::string(input));
    dim3 block_size(1, 1);
    dim3 grid_size(1);

    unsigned int *d_s;
    cudaMalloc(&d_s, sizeof(unsigned int) * 8);

    sha256 << <1, 1 >> > (d_s);
    cudaDeviceSynchronize();

    unsigned int h_s[8] = { 0 };
    cudaMemcpy(h_s, d_s, sizeof(unsigned int) * 8, cudaMemcpyDeviceToHost);
    cudaFree(d_s);

    std::stringstream stream;
    for (unsigned int i = 0; i < 8; ++i) {
        stream << std::hex << std::setfill('0') << std::setw(8) << h_s[i];
    }
    char *ch = new char[stream.str().length() + 1];
    strcpy(ch, stream.str().c_str());
    return ch;
}