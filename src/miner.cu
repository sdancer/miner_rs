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

// custom memcpy, apparently cuda's memcpy is slow 
// when called within a kernel
__device__ void g_memcpy(u32 *lhs, const u32 *rhs, int size) {
    // assuming u32 is 4 bytes
    int len = size / 4;
    for(int i=0; i<len; i++)
        lhs[i] = rhs[i];
}

// custom memset
template<typename T, typename ptr_t>
__device__ void g_memset(ptr_t dest, T val, int count) {
    for(int i=0; i<count; i++)
        dest[i] = val;
}

__constant__ const int g_MSG_PERMUTATION[] = {
    2, 6, 3, 10, 7, 0, 4, 13, 
    1, 11, 12, 5, 9, 14, 15, 8
};

__device__ u32 g_rotr(u32 value, int shift) {
    return (value >> shift)|(value << (usize - shift));
}

__device__ void g_g(u32 state[16], u32 a, u32 b, u32 c, u32 d, u32 mx, u32 my) {
    state[a] = state[a] + state[b] + mx;
    state[d] = g_rotr((state[d] ^ state[a]), 16);
    state[c] = state[c] + state[d];

    state[b] = g_rotr((state[b] ^ state[c]), 12);
    state[a] = state[a] + state[b] + my;
    state[d] = g_rotr((state[d] ^ state[a]), 8);

    state[c] = state[c] + state[d];
    state[b] = g_rotr((state[b] ^ state[c]), 7);
}

__device__ void g_round(u32 state[16], u32 m[16]) {
    // Mix the columns.
    g_g(state, 0, 4, 8, 12, m[0], m[1]);
    g_g(state, 1, 5, 9, 13, m[2], m[3]);
    g_g(state, 2, 6, 10, 14, m[4], m[5]);
    g_g(state, 3, 7, 11, 15, m[6], m[7]);
    // Mix the diagonals.
    g_g(state, 0, 5, 10, 15, m[8], m[9]);
    g_g(state, 1, 6, 11, 12, m[10], m[11]);
    g_g(state, 2, 7, 8, 13, m[12], m[13]);
    g_g(state, 3, 4, 9, 14, m[14], m[15]);
}

__device__ void g_permute(u32 m[16]) {
    u32 permuted[16];
    for(int i=0; i<16; i++)
        permuted[i] = m[g_MSG_PERMUTATION[i]];
    for(int i=0; i<16; i++)
        m[i] = permuted[i];
}

__device__ void g_compress(
    u32 *chaining_value,
    u32 *block_words,
    u64 counter,
    u32 block_len,
    u32 flags,
    u32 *state
) {
    // Search for better alternative
    g_memcpy(state, chaining_value, 32);
    g_memcpy(state+8, g_IV, 16);
    state[12] = (u32)counter;
    state[13] = (u32)(counter >> 32);
    state[14] = block_len;
    state[15] = flags;
    printf("counter %lx", counter);

    u32 block[16];
    g_memcpy(block, block_words, 64);
    
    g_round(state, block); // round 1
    g_permute(block);
    g_round(state, block); // round 2
    g_permute(block);
    g_round(state, block); // round 3
    g_permute(block);
    g_round(state, block); // round 4
    g_permute(block);
    g_round(state, block); // round 5
    g_permute(block);
    g_round(state, block); // round 6
    g_permute(block);
    g_round(state, block); // round 7

    for(int i=0; i<8; i++){
        state[i] ^= state[i + 8];
        state[i + 8] ^= chaining_value[i];
    }
}



extern "C" __global__ void compress(
    u32 *__restrict__ chaining_value,  // cv[8]
    u32 *__restrict__ block_words,     // m[16]
    u64 counter,
    u32 block_len,
    u32 flags,
    u32 *__restrict__ state_out)             // writes v[16]
{
 
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW == 0 && COL == 0) {
      g_compress(chaining_value, block_words, counter, block_len, flags, state_out);
      printf("got called %lx\n",state_out[0]);
    }
}

