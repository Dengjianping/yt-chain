#include "sha.h"


#define rotate_right(a,b) (((a) >> (b)) | ((a) << (64-(b))))
#define ch(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define maj(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define sigma_0(x) (rotate_right(x,28) ^ rotate_right(x,34) ^ rotate_right(x,39))
#define sigma_1(x) (rotate_right(x,14) ^ rotate_right(x,18) ^ rotate_right(x,41))
#define ep_0(x) (rotate_right(x,1) ^ rotate_right(x,8) ^ ((x) >> 7))
#define ep_1(x) (rotate_right(x,19) ^ rotate_right(x,61) ^ ((x) >> 6))

using uint_64 = unsigned long long int;


__constant__ uint_64 H[8] = {
    0xcbbb9d5dc1059ed8,
    0x629a292a367cd507,
    0x9159015a3070dd17,
    0x152fecd8f70e5939,
    0x67332667ffc00b31,
    0x8eb44a8768581511,
    0xdb0c2e0d64f98fa7,
    0x47b5481dbefa4fa4
};

__constant__ uint_64 K[80] = {
    0x428A2F98D728AE22, 0x7137449123EF65CD, 0xB5C0FBCFEC4D3B2F, 0xE9B5DBA58189DBBC,
    0x3956C25BF348B538, 0x59F111F1B605D019, 0x923F82A4AF194F9B, 0xAB1C5ED5DA6D8118,
    0xD807AA98A3030242, 0x12835B0145706FBE, 0x243185BE4EE4B28C, 0x550C7DC3D5FFB4E2,
    0x72BE5D74F27B896F, 0x80DEB1FE3B1696B1, 0x9BDC06A725C71235, 0xC19BF174CF692694,
    0xE49B69C19EF14AD2, 0xEFBE4786384F25E3, 0x0FC19DC68B8CD5B5, 0x240CA1CC77AC9C65,
    0x2DE92C6F592B0275, 0x4A7484AA6EA6E483, 0x5CB0A9DCBD41FBD4, 0x76F988DA831153B5,
    0x983E5152EE66DFAB, 0xA831C66D2DB43210, 0xB00327C898FB213F, 0xBF597FC7BEEF0EE4,
    0xC6E00BF33DA88FC2, 0xD5A79147930AA725, 0x06CA6351E003826F, 0x142929670A0E6E70,
    0x27B70A8546D22FFC, 0x2E1B21385C26C926, 0x4D2C6DFC5AC42AED, 0x53380D139D95B3DF,
    0x650A73548BAF63DE, 0x766A0ABB3C77B2A8, 0x81C2C92E47EDAEE6, 0x92722C851482353B,
    0xA2BFE8A14CF10364, 0xA81A664BBC423001, 0xC24B8B70D0F89791, 0xC76C51A30654BE30,
    0xD192E819D6EF5218, 0xD69906245565A910, 0xF40E35855771202A, 0x106AA07032BBD1B8,
    0x19A4C116B8D2D0C8, 0x1E376C085141AB53, 0x2748774CDF8EEB99, 0x34B0BCB5E19B48A8,
    0x391C0CB3C5C95A63, 0x4ED8AA4AE3418ACB, 0x5B9CCA4F7763E373, 0x682E6FF3D6B2B8A3,
    0x748F82EE5DEFB2FC, 0x78A5636F43172F60, 0x84C87814A1F0AB72, 0x8CC702081A6439EC,
    0x90BEFFFA23631E28, 0xA4506CEBDE82BDE9, 0xBEF9A3F7B2C67915, 0xC67178F2E372532B,
    0xCA273ECEEA26619C, 0xD186B8C721C0C207, 0xEADA7DD6CDE0EB1E, 0xF57D4F7FEE6ED178,
    0x06F067AA72176FBA, 0x0A637DC5A2C898A6, 0x113F9804BEF90DAE, 0x1B710B35131C471B,
    0x28DB77F523047D84, 0x32CAAB7B40C72493, 0x3C9EBE0A15C9BEBC, 0x431D67C49C100D4C,
    0x4CC5D4BECB3E42B6, 0x597F299CFC657E2A, 0x5FCB6FAB3AD6FAEC, 0x6C44198C4A475817
};


__global__ void sha384(const uint_64* __restrict__ d_input, uint_64* d_output, unsigned int length) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    uint_64 temp[80];
    uint_64 h[8] = {
		0xcbbb9d5dc1059ed8,
		0x629a292a367cd507,
		0x9159015a3070dd17,
		0x152fecd8f70e5939,
		0x67332667ffc00b31,
		0x8eb44a8768581511,
		0xdb0c2e0d64f98fa7,
		0x47b5481dbefa4fa4
	};
	#pragma unroll
    for (unsigned int i = 0; i < 16; ++i) temp[i] = d_input[length * i + index];

	#pragma unroll
    for (unsigned int i = 16; i < 80; ++i) {
        temp[i] = ep_1(temp[i - 2]) + temp[i - 7] + ep_0(temp[i - 15]) + temp[i - 16];
    }

	#pragma unroll
    for (unsigned int i = 0; i < 80; ++i) {
        uint_64 z_d = h[7] + temp[i] + K[i] + ch(h[4], h[5], h[6]) + sigma_1(h[4]);
        uint_64 z_a = maj(h[0], h[1], h[2]) + sigma_0(h[0]);

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
	for (unsigned int i = 0; i < 6; ++i) {
		d_output[length * i + index] = h[i] + H[i];
	}
}


void paddle_bits_384(const std::string* _input, int elements_num, uint_64* d_input) {
	uint_64* unaligned = new uint_64[16 * elements_num];
	uint_64* aligned = new uint_64[16 * elements_num];

    for (auto i = 0; i < elements_num; ++i) {
    	std::string c = "";
        for (auto j = 0; j < _input[i].length(); ++j) {
            c += std::bitset<8>(_input[i][j]).to_string();
        }

        c += "10000000"; // 0x80
        int len = c.length();
        c += std::string(896 - len, '0');
        std::string j = std::bitset<8>(_input[i].length() * 8).to_string();
        c += std::string(128 - j.length(), '0');
        c += j;

        for (auto j = 0; j < 16; ++j) {
        	unaligned[16 * i + j] = std::stoull(c.substr(64 * j, 64), nullptr, 2); // stride is 32
        }
    }

    for (auto i = 0; i < 16; ++i) {
        for (auto j = 0; j < elements_num; ++j) {
            // re-arrange data to avoid not aligning access global data
            aligned[elements_num * i + j] = unaligned[16 * j + i];
        }
    }

    cudaMemcpy(d_input, aligned, sizeof(uint_64) * 16 * elements_num, cudaMemcpyHostToDevice);
    delete[] unaligned;
    delete[] aligned;
}

extern "C" const uint_64  SHA384(const uint_64 prev_proof, const char *proof_of_work) {
	cudaDeviceProp device_prop;
	cudaGetDeviceProperties(&device_prop, 0);

	int threads_per_block = device_prop.maxThreadsPerMultiProcessor / device_prop.warpSize;
	// hash how many elements each time
	const int hash_elements = device_prop.maxThreadsPerMultiProcessor * device_prop.multiProcessorCount * 50;

    std::string* un_paddled = new std::string[hash_elements];
    uint_64* h_result = new uint_64[6 * hash_elements];

    uint_64 *d_input;
    cudaMalloc(&d_input, sizeof(uint_64) * 16 * hash_elements);

    uint_64 *d_output;
    cudaMalloc(&d_output, sizeof(uint_64) * 6 *hash_elements);

    dim3 block_size(threads_per_block, 1);
    dim3 grid_size(hash_elements / threads_per_block, 1);

    uint_64 caculated_num = 0;
    uint_64 proof = 0;
    while (true) {
		bool found = false;
    	for (auto i = caculated_num; i < caculated_num + hash_elements; ++i) {
    		un_paddled[i - caculated_num] = std::to_string(prev_proof) + std::to_string(i);
		}

    	paddle_bits_384(un_paddled, hash_elements, d_input);

    	sha384 <<<grid_size, block_size >> > (d_input, d_output, hash_elements);
    	cudaDeviceSynchronize();

		cudaMemcpy(h_result, d_output, sizeof(uint_64) * 6 * hash_elements, cudaMemcpyDeviceToHost);

		for (auto i = 0; i < hash_elements; ++i) {
			std::stringstream stream;
			for (auto j = 0; j < 6; ++j) {
				stream << std::hex << std::setfill('0') << std::setw(16) << h_result[hash_elements * j + i];
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