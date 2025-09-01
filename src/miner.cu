typedef unsigned long long uint64_t;
typedef unsigned int  uint32_t;
typedef unsigned char uint8_t;

using u32 = uint32_t;
using u64 = uint64_t;
using u8  = uint8_t;
 
const u32 OUT_LEN = 32;
const u32 KEY_LEN = 32;
const u32 BLOCK_LEN = 64;
const u32 CHUNK_LEN = 1024;
// Multiple chunks make a snicker bar :)
const u32 SNICKER = 1U << 10;
// Factory height and snicker size have an inversly propotional relationship
// FACTORY_HT * (log2 SNICKER) + 10 >= 64 
const u32 FACTORY_HT = 5;

const u32 CHUNK_START = 1 << 0;
const u32 CHUNK_END = 1 << 1;
const u32 PARENT = 1 << 2;
const u32 ROOT = 1 << 3;
const u32 KEYED_HASH = 1 << 4;
const u32 DERIVE_KEY_CONTEXT = 1 << 5;
const u32 DERIVE_KEY_MATERIAL = 1 << 6;

const int usize = sizeof(u32) * 8;

// redefine functions, but for the GPU
// all of them are the same but with g_ prefixed
__constant__ const u32 g_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
};

extern "C" __global__ void compress(u8* A, u8 * C) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW == 0 && COL == 0) {
      printf("got called\n");
 
    }
}

extern "C" __global__ void matmul(float* A, float* B, float* C, int N) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    // printf(\"pos, (%d, %d) - N %d - value %d\\n\", ROW, COL, N, tmpSum);
    C[ROW * N + COL] = tmpSum;
}
