/*
the reference: https://en.wikipedia.org/wiki/MD5
*/

#include "sha.h"


__inline__ __device__ unsigned int rotate_right(unsigned int a, unsigned int b) {
	return __funnelshift_r(a, a, b);
}

__inline__ __device__ unsigned int ch(unsigned int x, unsigned int y, unsigned int z) {
	return (x & y) ^ (~x & z);
}

__inline__ __device__ unsigned int maj(unsigned int x, unsigned int y, unsigned int z) {
	return (x & y) ^ (x & z) ^ (y & z);
}

__inline__ __device__ unsigned int sigma_0(unsigned int x) {
	return rotate_right(x, 2) ^ rotate_right(x, 13) ^ rotate_right(x, 22);
}

__inline__ __device__ unsigned int sigma_1(unsigned int x) {
	return rotate_right(x, 6) ^ rotate_right(x, 11) ^ rotate_right(x, 25);
}

__inline__ __device__ unsigned int ep_0(unsigned int x) {
	return rotate_right(x, 7) ^ rotate_right(x, 18) ^ (x >> 3);
}

__inline__ __device__ unsigned int ep_1(unsigned int x) {
	return rotate_right(x, 17) ^ rotate_right(x, 19) ^ (x >> 10);
}

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


__constant__ unsigned int K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};


__global__ void sha256(const unsigned int* __restrict__ d_input, unsigned int* __restrict__ d_output, unsigned int length) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int temp[64];

    unsigned int h[8] = {
        0x6a09e667,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19
    };

    if (index < length) {
		#pragma unroll
		for (unsigned int i = 0; i < 16; ++i) temp[i] = d_input[length * i + index];

		#pragma unroll
		for (unsigned int i = 16; i < 64; ++i) {
			temp[i] = ep_1(temp[i - 2]) + temp[i - 7] + ep_0(temp[i - 15]) + temp[i - 16];
		}

		#pragma unroll
		for (unsigned int i = 0; i < 64; ++i) {
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
			d_output[length * i + index] = h[i] + H[i];
		}
    }
}

void paddle_bits_256(const std::string* _input, int elements_num, unsigned int* d_input) {
    unsigned int* unaligned = new unsigned int[16 * elements_num];
    unsigned int* aligned = new unsigned int[16 * elements_num];

    for (unsigned int i = 0; i < elements_num; ++i) {
    	std::string c = "";
        for (unsigned int j = 0; j < _input[i].length(); ++j) {
            c += std::bitset<8>(_input[i][j]).to_string();
        }

        c += "10000000"; // 0x80
        int len = c.length();
        c += std::string(448 - len, '0');
        std::string j = std::bitset<8>(_input[i].length() * 8).to_string();
        c += std::string(64 - j.length(), '0');
        c += j;

        for (unsigned int j = 0; j < 16; ++j) {
        	unaligned[16 * i + j] = std::stoll(c.substr(32 * j, 32), nullptr, 2); // stride is 32
        }
    }

    for (auto i = 0; i < 16; ++i) {
        for (auto j = 0; j < elements_num; ++j) {
            // re-arrange data to avoid not aligning access global data
            aligned[elements_num * i + j] = unaligned[16 * j + i];
        }
    }

    cudaMemcpy(d_input, aligned, sizeof(unsigned int) * 16 * elements_num, cudaMemcpyHostToDevice);
    delete[] unaligned;
    delete[] aligned;
}


extern "C" const unsigned int  SHA256(const unsigned int prev_proof, const char *proof_of_work) {
	cudaDeviceProp device_prop;
	cudaGetDeviceProperties(&device_prop, 0);

	int threads_per_block = device_prop.maxThreadsPerMultiProcessor / device_prop.warpSize;
	// hash how many elements each time
	const int hash_elements = device_prop.maxThreadsPerMultiProcessor * device_prop.multiProcessorCount * 50;

    std::string* un_paddled = new std::string[hash_elements];
    unsigned int* h_result = new unsigned int[8 * hash_elements];

    unsigned int *d_input;
    cudaMalloc(&d_input, sizeof(unsigned int) * 16 * hash_elements);

    unsigned int *d_output;
    cudaMalloc(&d_output, sizeof(unsigned int) * 8 *hash_elements);

    dim3 block_size(threads_per_block, 1);
    dim3 grid_size(hash_elements / threads_per_block, 1);

    unsigned int caculated_num = 0;
    unsigned int proof = 0;
    while (true) {
		bool found = false;
    	for (unsigned int i = caculated_num; i < caculated_num + hash_elements; ++i) {
    		un_paddled[i - caculated_num] = std::to_string(prev_proof) + std::to_string(i);
		}

    	paddle_bits_256(un_paddled, hash_elements, d_input);

    	sha256 <<<grid_size, block_size >> > (d_input, d_output, hash_elements);
    	cudaDeviceSynchronize();

		cudaMemcpy(h_result, d_output, sizeof(unsigned int) * 8 * hash_elements, cudaMemcpyDeviceToHost);

		for (auto i = 0; i < hash_elements; ++i) {
			std::stringstream stream;
			for (auto j = 0; j < 8; ++j) {
				stream << std::hex << std::setfill('0') << std::setw(8) << h_result[hash_elements * j + i];
			}
			if (stream.str().rfind(proof_of_work, 0) == 0) {
				proof = caculated_num + i;
				caculated_num = 0;
				found = true;
				break;
			}
		}
		if (found == true) {
			break;
		}
		caculated_num += hash_elements;
	}

    delete[] h_result;
    cudaFree(d_output);
    cudaFree(d_input);

    return proof;
}