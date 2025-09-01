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

struct Chunk {
    // use only when it is a leaf node
    // leaf data may have less than 1024 bytes
    u8 leaf_data[1024];
    u32 leaf_len;
    // use in all other cases
    // data will always have 64 bytes
    u32 data[16];
    u32 flags;
    u32 raw_hash[16];
    u32 key[8];
    // only useful for leaf nodes
    u64 counter;
    // Constructor for leaf nodes
    //Chunk(char *input, int size, u32 _flags, u32 *_key, u64 ctr){
    //    counter = ctr;
    //    flags = _flags;
    //    memcpy(key, _key, 8*sizeof(*key));
    //    memset(leaf_data, 0, 1024);
    //    memcpy(leaf_data, input, size);
    //    leaf_len = size;
    //}
    //Chunk(u32 _flags, u32 *_key) {
    //    counter = 0;
    //    flags = _flags;
    //    memcpy(key, _key, 8*sizeof(*key));
    //    leaf_len = 0;
    //}
    //Chunk() {}
    // Chunk() : leaf_len(0) {}
    // process data in sizes of message blocks and store cv in hash
    //void compress_chunk(u32=0);
    //__device__ void g_compress_chunk(u32=0);
};

// redefine functions, but for the GPU
// all of them are the same but with g_ prefixed
__constant__ const u32 g_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
};

__constant__ const int g_MSG_PERMUTATION[] = {
    2, 6, 3, 10, 7, 0, 4, 13, 
    1, 11, 12, 5, 9, 14, 15, 8
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

__device__ __forceinline__ uint32_t g_rotr32(uint32_t v, int s) {
    return (v >> s) | (v << (32 - s));
}

#define GG(a,b,c,d, mx,my)         \
    do {                           \
        (a) += (b) + (mx);         \
        (d)  = g_rotr32((d) ^ (a), 16); \
        (c) += (d);                \
        (b)  = g_rotr32((b) ^ (c), 12); \
        (a) += (b) + (my);         \
        (d)  = g_rotr32((d) ^ (a), 8);  \
        (c) += (d);                \
        (b)  = g_rotr32((b) ^ (c), 7);  \
    } while (0)

// m can be passed as separate regs (m0..m15) or you can load them from memory first.
// This keeps *state* entirely in registers via references.
__device__ __forceinline__ void g_round_regs(
    uint32_t &s0,  uint32_t &s1,  uint32_t &s2,  uint32_t &s3,
    uint32_t &s4,  uint32_t &s5,  uint32_t &s6,  uint32_t &s7,
    uint32_t &s8,  uint32_t &s9,  uint32_t &s10, uint32_t &s11,
    uint32_t &s12, uint32_t &s13, uint32_t &s14, uint32_t &s15,
    const uint32_t m0,  const uint32_t m1,  const uint32_t m2,  const uint32_t m3,
    const uint32_t m4,  const uint32_t m5,  const uint32_t m6,  const uint32_t m7,
    const uint32_t m8,  const uint32_t m9,  const uint32_t m10, const uint32_t m11,
    const uint32_t m12, const uint32_t m13, const uint32_t m14, const uint32_t m15)
{
    // Mix the columns.
    GG(s0,  s4,  s8,  s12, m0,  m1);
    GG(s1,  s5,  s9,  s13, m2,  m3);
    GG(s2,  s6,  s10, s14, m4,  m5);
    GG(s3,  s7,  s11, s15, m6,  m7);
    // Mix the diagonals.
    GG(s0,  s5,  s10, s15, m8,  m9);
    GG(s1,  s6,  s11, s12, m10, m11);
    GG(s2,  s7,  s8,  s13, m12, m13);
    GG(s3,  s4,  s9,  s14, m14, m15);
}

__device__ __forceinline__ void g_permute_regs(
    u32 &m0,  u32 &m1,  u32 &m2,  u32 &m3,
    u32 &m4,  u32 &m5,  u32 &m6,  u32 &m7,
    u32 &m8,  u32 &m9,  u32 &m10, u32 &m11,
    u32 &m12, u32 &m13, u32 &m14, u32 &m15)
{
    // BLAKE3 message permutation:
    // [2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]
    u32 t0=m0, t1=m1, t2=m2, t3=m3, t4=m4, t5=m5, t6=m6, t7=m7;
    u32 t8=m8, t9=m9, t10=m10, t11=m11, t12=m12, t13=m13, t14=m14, t15=m15;

    m0  = t2;   m1  = t6;   m2  = t3;   m3  = t10;
    m4  = t7;   m5  = t0;   m6  = t4;   m7  = t13;
    m8  = t1;   m9  = t11;  m10 = t12;  m11 = t5;
    m12 = t9;   m13 = t14;  m14 = t15;  m15 = t8;
}

__device__ __forceinline__ void g_compress(
    const u32 *__restrict__ chaining_value,  // cv[8]
    const u32 *__restrict__ block_words,     // m[16]
    u64 counter,
    u32 block_len,
    u32 flags,
    u32 *__restrict__ state_out)             // writes v[16]
{
    // ---- Load CV into regs (keep originals for feedforward) ----
    u32 cv0 = chaining_value[0], cv1 = chaining_value[1];
    u32 cv2 = chaining_value[2], cv3 = chaining_value[3];
    u32 cv4 = chaining_value[4], cv5 = chaining_value[5];
    u32 cv6 = chaining_value[6], cv7 = chaining_value[7];

    // Working state in 16 registers
    u32 s0=cv0, s1=cv1, s2=cv2, s3=cv3, s4=cv4, s5=cv5, s6=cv6, s7=cv7;
    u32 s8=g_IV[0], s9=g_IV[1], s10=g_IV[2], s11=g_IV[3];
    u32 s12=(u32)counter, s13=(u32)(counter >> 32), s14=block_len, s15=flags;

    // ---- Load message into regs ----
    u32 m0 = block_words[0],  m1  = block_words[1];
    u32 m2 = block_words[2],  m3  = block_words[3];
    u32 m4 = block_words[4],  m5  = block_words[5];
    u32 m6 = block_words[6],  m7  = block_words[7];
    u32 m8 = block_words[8],  m9  = block_words[9];
    u32 m10= block_words[10], m11 = block_words[11];
    u32 m12= block_words[12], m13 = block_words[13];
    u32 m14= block_words[14], m15 = block_words[15];

    // ---- 7 rounds (permute between rounds, not after the last) ----
    g_round_regs(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
                 m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15);

    g_permute_regs(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15);
    g_round_regs(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
                 m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15);

    g_permute_regs(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15);
    g_round_regs(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
                 m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15);

    g_permute_regs(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15);
    g_round_regs(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
                 m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15);

    g_permute_regs(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15);
    g_round_regs(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
                 m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15);

    g_permute_regs(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15);
    g_round_regs(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
                 m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15);

    g_permute_regs(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15);
    g_round_regs(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
                 m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15);

    // ---- Feedforward / output transform ----
    s0  ^= s8;   s1  ^= s9;   s2  ^= s10;  s3  ^= s11;
    s4  ^= s12;  s5  ^= s13;  s6  ^= s14;  s7  ^= s15;

    s8  ^= cv0;  s9  ^= cv1;  s10 ^= cv2;  s11 ^= cv3;
    s12 ^= cv4;  s13 ^= cv5;  s14 ^= cv6;  s15 ^= cv7;

    // ---- Write out ----
    state_out[0]=s0;  state_out[1]=s1;   state_out[2]=s2;   state_out[3]=s3;
    state_out[4]=s4;  state_out[5]=s5;   state_out[6]=s6;   state_out[7]=s7;
    state_out[8]=s8;  state_out[9]=s9;   state_out[10]=s10; state_out[11]=s11;
    state_out[12]=s12;state_out[13]=s13; state_out[14]=s14; state_out[15]=s15;
}

__device__ void g_words_from_little_endian_bytes(
    u8 *bytes, u32 *words, u32 bytes_len
) {
    u32 tmp;
    for(u32 i=0; i<bytes_len; i+=4) {
        tmp = (bytes[i+3]<<24) | (bytes[i+2]<<16) | (bytes[i+1]<<8) | bytes[i];
        words[i/4] = tmp;
    }
}

extern "C" __global__ void compress(uint8* A, uint8* C) {
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW == 0 && COL == 0) {
      //printf("got called\n");
 
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
