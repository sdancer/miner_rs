#include <sm_61_intrinsics.h>

typedef unsigned long long uint64_t;
typedef unsigned int  uint32_t;
typedef unsigned char uint8_t;

typedef signed int  int32_t;
typedef signed char int8_t;

using u32 = uint32_t;
using u64 = uint64_t;
using u8  = uint8_t;
 
#include <mma.h>
using namespace nvcuda;
namespace wexp = nvcuda::wmma::experimental;

//const u32 OUT_LEN = 32;
//const u32 KEY_LEN = 32;
//const u32 BLOCK_LEN = 64;
//const u32 CHUNK_LEN = 1024;
//// Multiple chunks make a snicker bar :)
//const u32 SNICKER = 1U << 10;
//// Factory height and snicker size have an inversly propotional relationship
//// FACTORY_HT * (log2 SNICKER) + 10 >= 64 
//const u32 FACTORY_HT = 5;
//
const u32 CHUNK_START = 1 << 0;
const u32 CHUNK_END = 1 << 1;
//const u32 PARENT = 1 << 2;
const u32 ROOT = 1 << 3;
//const u32 KEYED_HASH = 1 << 4;
//const u32 DERIVE_KEY_CONTEXT = 1 << 5;
//const u32 DERIVE_KEY_MATERIAL = 1 << 6;

//const int usize = sizeof(u32) * 8;

// redefine functions, but for the GPU
// all of them are the same but with g_ prefixed
__constant__ const u32 g_IV[8] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
};


#ifdef NEWGCOMP
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

__device__ __forceinline__ void g_compress(
    const u32 *__restrict__ chaining_value,  // cv[8]
    const u32 *__restrict__ block_words,     // m[16]
    u32 counter,
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
    u32 s12=(u32)counter, s13=0, s14=block_len, s15=flags;

    // ---- Load message into regs ----
    u32 m0 = block_words[0],  m1  = block_words[1];
    u32 m2 = block_words[2],  m3  = block_words[3];
    u32 m4 = block_words[4],  m5  = block_words[5];
    u32 m6 = block_words[6],  m7  = block_words[7];
    u32 m8 = block_words[8],  m9  = block_words[9];
    u32 m10= block_words[10], m11 = block_words[11];
    u32 m12= block_words[12], m13 = block_words[13];
    u32 m14= block_words[14], m15 = block_words[15];

    // Round 0: identity
    g_round_regs(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
                 m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15);
    
    // Round 1: perm^1 = [2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]
    g_round_regs(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
                 m2,m6,m3,m10,m7,m0,m4,m13,m1,m11,m12,m5,m9,m14,m15,m8);
    
    // Round 2: perm^2 = [3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1]
    g_round_regs(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
                 m3,m4,m10,m12,m13,m2,m7,m14,m6,m5,m9,m0,m11,m15,m8,m1);
    
    // Round 3: perm^3 = [10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6]
    g_round_regs(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
                 m10,m7,m12,m9,m14,m3,m13,m15,m4,m0,m11,m2,m5,m8,m1,m6);
    
    // Round 4: perm^4 = [12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4]
    g_round_regs(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
                 m12,m13,m9,m11,m15,m10,m14,m8,m7,m2,m5,m3,m0,m1,m6,m4);
    
    // Round 5: perm^5 = [9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7]
    g_round_regs(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
                 m9,m14,m11,m5,m8,m12,m15,m1,m13,m3,m0,m10,m2,m6,m4,m7);
    
    // Round 6: perm^6 = [11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13]
    g_round_regs(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,
                 m11,m15,m5,m0,m1,m9,m8,m6,m14,m10,m2,m12,m3,m4,m7,m13);

    // ---- Feedforward / output transform ----
    s0  ^= s8;   s1  ^= s9;   s2  ^= s10;  s3  ^= s11;
    s4  ^= s12;  s5  ^= s13;  s6  ^= s14;  s7  ^= s15;

    // ---- Write out ----
    state_out[0]=s0;  state_out[1]=s1;   state_out[2]=s2;   state_out[3]=s3;
    state_out[4]=s4;  state_out[5]=s5;   state_out[6]=s6;   state_out[7]=s7;
    state_out[8]=s8;  state_out[9]=s9;   state_out[10]=s10; state_out[11]=s11;
    state_out[12]=s12;state_out[13]=s13; state_out[14]=s14; state_out[15]=s15;
}
#else
__constant__ const int g_MSG_PERMUTATION[] = {
    2, 6, 3, 10, 7, 0, 4, 13,
    1, 11, 12, 5, 9, 14, 15, 8
};

__device__ __forceinline__ u32 g_rotr(u32 value, int shift) {
    // OPTIMIZATION: Use fast bit rotation with compiler intrinsics for lower precision/higher speed
    return __funnelshift_r(value, value, shift);
}

__device__ __forceinline__ void g_g(u32 state[16], u32 a, u32 b, u32 c, u32 d, u32 mx, u32 my) {
    // OPTIMIZATION: Fast arithmetic with reduced precision (CPU verification will catch errors)
    // Use fast unchecked arithmetic - overflow is acceptable for speed
    u32 temp_a = state[a] + state[b] + mx; // Fast unchecked addition
    state[d] = g_rotr((state[d] ^ temp_a), 16);
    u32 temp_c = state[c] + state[d];

    state[b] = g_rotr((state[b] ^ temp_c), 12);
    temp_a += state[b] + my; // Fast unchecked addition
    state[d] = g_rotr((state[d] ^ temp_a), 8);

    temp_c += state[d]; // Fast unchecked addition
    state[b] = g_rotr((state[b] ^ temp_c), 7);

    // Write back results
    state[a] = temp_a;
    state[c] = temp_c;
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

__device__ __forceinline__ void g_compress(
    const u32 *chaining_value,
    const u32 *block_words,
    u64 counter,
    u32 block_len,
    u32 flags,
    u32 *state
) {

    g_memcpy(state, chaining_value, 32);
    g_memcpy(state+8, g_IV, 16);
    state[12] = (u32)counter;
    state[13] = (u32)(counter >> 32);
    state[14] = block_len;
    state[15] = flags;

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


    for(int i = 0; i < 8; i++){
        state[i] ^= state[i + 8];
    }
}
#endif


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
      g_compress(chaining_value, block_words, counter & 0xffffffff, block_len, flags, state_out);
 //     printf("got called %lx\n",state_out[0]);
    }
}


// --- Emit one 64B XOF block into 16 u32 words (dstw)
__device__ inline void xof_emit_words(
    u32 blk,
    const u32 root[8],
    const u32 precv[8],
    const u32 last_words[16],
    u32 last_len,
    u32 dstw[16])
{
    u32 out[16];

    const u32 t = blk;
    //printf("blk: %x\n", blk);

    const u32 flags = (CHUNK_END | ROOT);
    // g_compress writes 16 words to state; its low half is already lo^hi (root CV),
    // high half is the raw hi (no feed-forward).
    g_compress(precv, const_cast<u32*>(last_words), t, last_len, flags, out);

    #pragma unroll
    for (int w=0; w<8; ++w) dstw[w] = out[w];

    #pragma unroll
    for (int w=0; w<8; ++w) dstw[8+w] = out[8+w] ^ precv[w];
}

#ifndef TILE_K
#define TILE_K 256  // multiple of 64 and 4
#endif


// Helpers
__device__ __forceinline__ void store_le64(uint8_t* dst, u64 x) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) dst[i] = (uint8_t)(x >> (8*i));
}

// Compute root + pre-final CV + last words from a 240B seed
__device__ __forceinline__
void compute_root_from_seed240(const uint8_t* __restrict__ seed240,
                               u32* __restrict__ out_root,   // 8
                               u32* __restrict__ out_precv,  // 8
                               u32* __restrict__ out_lwords, // 16
                               uint8_t* __restrict__ out_llen)
{
    u32 cv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = g_IV[i];

    u32 m[16], st[16];

    for (int blk = 0; blk < 4; ++blk) {
        const u32 blen = (blk == 3 ? 48u : 64u);
        // pack 64B (or 48B final) into 16 u32 (zero padded)
        #pragma unroll
        for (int w = 0; w < 16; ++w) m[w] = 0u;
        #pragma unroll
        for (u32 i = 0; i < blen; ++i)
            reinterpret_cast<uint8_t*>(m)[i] = seed240[blk*64 + i];

        u32 flags = 0;
        if (blk == 0) flags |= CHUNK_START;
        if (blk == 3) flags |= (CHUNK_END | ROOT);

        if (blk == 3) {
            #pragma unroll
            for (int w = 0; w < 8;  ++w) out_precv[w]  = cv[w];
            #pragma unroll
            for (int w = 0; w < 16; ++w) out_lwords[w] = m[w];
            *out_llen = (uint8_t)blen; // 48
        }

        g_compress(cv, m, 0ULL, blen, flags, st);
        #pragma unroll
        for (int w = 0; w < 8; ++w) cv[w] = st[w];
    }

    #pragma unroll
    for (int w = 0; w < 8; ++w) out_root[w] = cv[w];
}

// -----------------------------------------------------------------------------
// One kernel does everything for a range of nonces.
// Grid:  grid.x = #seeds (or any >=1, kernel loops by stride), block = (16,16)
// Smem:  dynamic = (16*TILE_K + TILE_K*16) bytes
// Args:
//   d_prefix232 : same 232B prefix for all seeds
//   nonce_start : starting 64-bit nonce (little-endian written into bytes 232..239)
//   nonce_count : number of seeds (one 16x16 output per nonce)
//   d_C         : outputs [nonce_count][16][16] int32
// -----------------------------------------------------------------------------
#ifndef TILE_K
#define TILE_K 256
#endif



// Simple cp.async wrapper: copy 16B from gmem to smem (assumes aligned)
__device__ __forceinline__ void cp_async_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));          // 32-bit
    unsigned long long gmem = (unsigned long long)__cvta_generic_to_global(gmem_ptr);   // 64-bit
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(gmem));
}


// ====== helpers (place near the top, reuse your g_compress/xof helpers) ======
__device__ __forceinline__ int pack4_sub128(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
    char4 v;
    v.x = (char)((int)b0 - 128);
    v.y = (char)((int)b1 - 128);
    v.z = (char)((int)b2 - 128);
    v.w = (char)((int)b3 - 128);
    return *reinterpret_cast<int*>(&v);
}

__device__ __forceinline__ int pack4_i8(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
    char4 v; v.x = (char)b0; v.y = (char)b1; v.z = (char)b2; v.w = (char)b3;
    return *reinterpret_cast<int*>(&v);
}

#define TILE_K 1024   // 4090: 32*TILE_K bytes of smem; 1024 => 32 KiB per block (comfortable)

// ================================== KERNEL ===================================
extern "C" __global__
__launch_bounds__(256, 2)
void solve_nonce_range_fused(
        const uint8_t* __restrict__ d_prefix232, // 232 bytes
        unsigned long long* d_iter_count,
        u64 nonce_start,
        int nonce_count,
        u32* __restrict__ d_hashes /* optional: debug slot */)
{
    const int i = threadIdx.y;   // 0..15
    const int j = threadIdx.x;   // 0..15
    if (i >= 16 || j >= 16) return;

    // ---- Small shared state (persists across tiles) ----
    __shared__ uint8_t sh_prefix[232];
    __shared__ uint8_t sh_seed[240];
    __shared__ u32 sh_root[8];
    __shared__ u32 sh_precv[8];
    __shared__ u32 sh_lwords[16];
    __shared__ uint8_t sh_llen; // 48 for the 4th block of 240B

    // Optional: keep C tile in shared (useful if you hash on-chip again)
    __shared__ int32_t tileC[16 * 16];

    if (i == 0 && j == 0) {
        #pragma unroll
        for (int t = 0; t < 232; ++t) sh_prefix[t] = d_prefix232[t];
    }
    __syncthreads();

    // ---- Dynamic shared (A_packed + small pad + B_packed) ----
    // Each pack is int32 where lanes are 4×int8.
    extern __shared__ __align__(16) uint8_t smem[];
    int32_t* As4 = reinterpret_cast<int32_t*>(smem);                                     // [16 rows][TILE_K/4]
    int32_t* Bs4 = reinterpret_cast<int32_t*>(smem + (size_t)16 * (TILE_K/4) * 4 + 64);  // [TILE_K/4 groups][16 cols]

    constexpr int K            = 50240;
    constexpr int A_BYTES      = 16 * K;           // 803,840
    constexpr int A_BLOCKS     = A_BYTES / 64;     // 12,560
    constexpr int B_BASE_BLOCK = A_BLOCKS;         // 12,560

    const int strideA = (TILE_K >> 2);   // int32 per row
    const int strideB = 16;              // int32 per k-group: one per column

    const int thread_id     = threadIdx.y * blockDim.x + threadIdx.x; // 0..255
    const int total_threads = blockDim.x * blockDim.y;

    for (int seed = blockIdx.x; seed < nonce_count; seed += gridDim.x) {
        if (i == 0 && j == 0) atomicAdd(d_iter_count, 1ULL);

        // Build seed & derive xof params once per seed
        if (i == 0 && j == 0) {
            #pragma unroll
            for (int t = 0; t < 232; ++t) sh_seed[t] = sh_prefix[t];
            const u64 nonce = nonce_start + (u64)seed;
            #pragma unroll
            for (int b = 0; b < 8; ++b) sh_seed[232 + b] = (uint8_t)(nonce >> (8*b));

            compute_root_from_seed240(sh_seed, sh_root, sh_precv, sh_lwords, &sh_llen);
        }
        __syncthreads();

        int acc   = 0;
        int sum_b = 0;                 // we’ll add (sum_b << 7) at the end
        const u32 llen = (u32)sh_llen;

        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            const int tile   = min(TILE_K, K - k0);
            const int groups = (tile + 3) >> 2;  // # of 4-k packs

            // -------------------- 1) PRE-PACK A (u8->i8, bias once) --------------------
            // A row r: produce (tile bytes) via XOF in 64B chunks → write as int32 packs.
            const int a_blocks_per_row = (tile + 63) / 64;
            for (int lin = thread_id; lin < 16 * a_blocks_per_row; lin += total_threads) {
                const int r       = lin / a_blocks_per_row;      // 0..15
                const int rb      = lin % a_blocks_per_row;      // 64B-block within row
                const int kk_base = rb * 64;
                if (kk_base >= tile) continue;

                u32 words[16];
                const uint32_t blkA = (uint32_t)(r * (K/64) + (k0/64) + rb);
                //xof_emit_words(blkA, sh_root, sh_precv, sh_lwords, llen, words);
                const uint8_t* w = reinterpret_cast<const uint8_t*>(words);

                #pragma unroll
                for (int q = 0; q < 16; ++q) {
                    const int gidx = (kk_base >> 2) + q;
                    if (gidx >= groups) break;

                    // guard tail per byte
                    const int b0k = kk_base + 4*q + 0;
                    const int b1k = kk_base + 4*q + 1;
                    const int b2k = kk_base + 4*q + 2;
                    const int b3k = kk_base + 4*q + 3;

                    uint8_t b0 = (b0k < tile) ? w[4*q + 0] : 0;
                    uint8_t b1 = (b1k < tile) ? w[4*q + 1] : 0;
                    uint8_t b2 = (b2k < tile) ? w[4*q + 2] : 0;
                    uint8_t b3 = (b3k < tile) ? w[4*q + 3] : 0;

                    As4[r * strideA + gidx] = pack4_sub128(b0, b1, b2, b3);
                }
            }

            // -------------------- 2) PRE-PACK B (transpose+pack by k-groups) -----------
            for (int gb = thread_id; gb < groups; gb += total_threads) {
                const int kk_base = gb * 4;
                const uint32_t blkB = (uint32_t)(B_BASE_BLOCK + ((k0 + kk_base) >> 2));
                u32 words[16];
                //xof_emit_words(blkB, sh_root, sh_precv, sh_lwords, llen, words);
                const uint8_t* w = reinterpret_cast<const uint8_t*>(words);

                // 64B is 4 “rows” of 16 bytes each → kk, kk+1, kk+2, kk+3
                #pragma unroll
                for (int jj = 0; jj < 16; ++jj) {
                    const bool k0ok = (kk_base + 0) < tile;
                    const bool k1ok = (kk_base + 1) < tile;
                    const bool k2ok = (kk_base + 2) < tile;
                    const bool k3ok = (kk_base + 3) < tile;

                    uint8_t b0 = k0ok ? w[0*16 + jj] : 0;
                    uint8_t b1 = k1ok ? w[1*16 + jj] : 0;
                    uint8_t b2 = k2ok ? w[2*16 + jj] : 0;
                    uint8_t b3 = k3ok ? w[3*16 + jj] : 0;

                    Bs4[gb * strideB + jj] = pack4_i8(b0, b1, b2, b3);
                }
            }
            __syncthreads();

            // -------------------- 3) COMPUTE (pure DP4A) ------------------------------
            const int ONES = 0x01010101;
            #pragma unroll 1
            for (int g = 0; g < groups; ++g) {
                const int a_p = As4[i * strideA + g];
                const int b_p = Bs4[g * strideB + j];
                acc   = __dp4a(a_p, b_p, acc);      // 4 MACs
                sum_b = __dp4a(ONES, b_p, sum_b);   // sum 4 int8s in one op
            }
            __syncthreads();
        }

        // Undo A’s (-128) bias: add 128 * sum_b
        acc += (sum_b << 7);

        tileC[i * 16 + j] = acc;
        __syncthreads();

        // tiny debug write to prove life
        if (i == 0 && j == 0 && seed == 0 && d_hashes) d_hashes[0] = (u32)tileC[0];
    }
}

