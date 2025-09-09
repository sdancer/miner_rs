#include <sm_61_intrinsics.h>

typedef unsigned long long uint64_t;
typedef unsigned int  uint32_t;
typedef unsigned char uint8_t;

typedef signed int  int32_t;
typedef signed char int8_t;

using u32 = uint32_t;
using u64 = uint64_t;
using u8  = uint8_t;
 

//const u32 OUT_LEN = 32;
//const u32 KEY_LEN = 32;
const u32 BLOCK_LEN = 64;
const u32 SEED_SIZE = 240;
//const u32 CHUNK_LEN = 1024;
//// Multiple chunks make a snicker bar :)
//const u32 SNICKER = 1U << 10;
//// Factory height and snicker size have an inversly propotional relationship
//// FACTORY_HT * (log2 SNICKER) + 10 >= 64 
//const u32 FACTORY_HT = 5;
//
const u32 CHUNK_START = 1 << 0;
const u32 CHUNK_END = 1 << 1;
const u32 PARENT = 1 << 2;
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


__device__
void fused_blake3_hash_and_detect(
    const uint8_t* __restrict__ d_seeds,   // 240B per item
    const int32_t* __restrict__ d_C,       // 256 i32 (1024B) per item
    uint64_t*  __restrict__ d_found_nonce,      
    uint32_t*  __restrict__ d_found_u32_at_228,
    uint64_t seed_n,

    unsigned long long* __restrict__ ring_nonces, // u64[cap]
    int                 ring_cap,                 // capacity
    int*                ring_flags,               // i32[cap], 0=empty, 1=full
    unsigned long long* __restrict__ ring_tail   // monotonically increasing reservation counter
)
{
    const int i = threadIdx.y;   // 0..15
    const int j = threadIdx.x;   // 0..15
    if (i == 0 && j == 0) {} else {
        return;
    }
 
    const uint8_t* seed = d_seeds;
    const uint8_t* Cb   = reinterpret_cast<const uint8_t*>(d_C);

    // ---- chunk 0 (1024B): seed[0..239] ++ Cb[0..783] ----
    u32 cv0[8];
    #pragma unroll
    for (int w=0; w<8; ++w) cv0[w] = g_IV[w];
    {
        u32 state[16];
        for (int blk=0; blk<16; ++blk) {
            u32 m[16] = {0};
            uint8_t* m_bytes = reinterpret_cast<uint8_t*>(m);

            // OPTIMIZATION: Vectorized memory operations for 64-byte blocks
            const int base_off = blk * 64;

            // Process in 16-byte chunks using uint4 vectorization
            #pragma unroll
            for (int chunk = 0; chunk < 4; ++chunk) {
                const int chunk_off = base_off + chunk * 16;
                uint4 data;

                if (chunk_off + 16 <= SEED_SIZE) {
                    // Fully within seed data - vectorized load
                    data = *reinterpret_cast<const uint4*>(seed + chunk_off);
                } else if (chunk_off >= SEED_SIZE) {
                    // Fully within Cb data - vectorized load
                    data = *reinterpret_cast<const uint4*>(Cb + (chunk_off - SEED_SIZE));
                } else {
                    // Boundary case - byte-by-byte (rare)
                    uint8_t temp[16];
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const int off = chunk_off + i;
                        temp[i] = (off < SEED_SIZE) ? seed[off] : Cb[off - SEED_SIZE];
                    }
                    data = *reinterpret_cast<const uint4*>(temp);
                }

                // Vectorized store
                *reinterpret_cast<uint4*>(m_bytes + chunk * 16) = data;
            }
            u32 flags = 0;
            if (blk == 0)  flags |= CHUNK_START;
            if (blk == 15) flags |= CHUNK_END;
            g_compress(cv0, m, /*t=*/0ULL, /*block_len=*/BLOCK_LEN, flags, state);
            #pragma unroll
            for (int w=0; w<8; ++w) cv0[w] = state[w];
        }
    }

    // ---- chunk 1 (240B): Cb[784..1023] ----
    u32 cv1[8];
    #pragma unroll
    for (int w=0; w<8; ++w) cv1[w] = g_IV[w];
    {
        u32 state[16];
        for (int blk=0; blk<4; ++blk) {
            const u32 blen = (blk == 3) ? 48u : 64u;
            u32 m[16] = {0};
            uint8_t* m_bytes = reinterpret_cast<uint8_t*>(m);
            const uint8_t* src = Cb + 784 + blk * 64;

            // OPTIMIZATION: Vectorized memory operations for chunk 1
            if (blen == 64u) {
                // Full 64-byte block - use 4x uint4 vectorized loads
                #pragma unroll
                for (int chunk = 0; chunk < 4; ++chunk) {
                    uint4 data = *reinterpret_cast<const uint4*>(src + chunk * 16);
                    *reinterpret_cast<uint4*>(m_bytes + chunk * 16) = data;
                }
            } else {
                // 48-byte block (last block) - use 3x uint4 + scalar
                #pragma unroll
                for (int chunk = 0; chunk < 3; ++chunk) {
                    uint4 data = *reinterpret_cast<const uint4*>(src + chunk * 16);
                    *reinterpret_cast<uint4*>(m_bytes + chunk * 16) = data;
                }
            }
            u32 flags = 0;
            if (blk == 0) flags |= CHUNK_START;
            if (blk == 3) flags |= CHUNK_END;
            g_compress(cv1, m, /*t=*/1ULL, blen, flags, state);
            #pragma unroll
            for (int w=0; w<8; ++w) cv1[w] = state[w];
        }
    }

    // ---- parent combine (ROOT) of the two leaf CVs ----
    u32 final_hash[8];
    {
        u32 parent_in[16];
        // OPTIMIZATION: Vectorized CV copying using uint4
        *reinterpret_cast<uint4*>(parent_in) = *reinterpret_cast<const uint4*>(cv0);
        *reinterpret_cast<uint4*>(parent_in + 4) = *reinterpret_cast<const uint4*>(cv0 + 4);
        *reinterpret_cast<uint4*>(parent_in + 8) = *reinterpret_cast<const uint4*>(cv1);
        *reinterpret_cast<uint4*>(parent_in + 12) = *reinterpret_cast<const uint4*>(cv1 + 4);

        u32 cv[8];
        // OPTIMIZATION: Vectorized IV initialization
        *reinterpret_cast<uint4*>(cv) = *reinterpret_cast<const uint4*>(g_IV);
        *reinterpret_cast<uint4*>(cv + 4) = *reinterpret_cast<const uint4*>(g_IV + 4);

        u32 st[16];
        g_compress(cv, parent_in, /*t=*/0ULL, /*block_len=*/BLOCK_LEN,
                   /*flags=*/PARENT | ROOT, st);

        // OPTIMIZATION: Vectorized final hash copying
        *reinterpret_cast<uint4*>(final_hash) = *reinterpret_cast<const uint4*>(st);
        *reinterpret_cast<uint4*>(final_hash + 4) = *reinterpret_cast<const uint4*>(st + 4);
    }

    // FUSION: Immediately check for solution without writing to global memory
    const uint8_t* h = reinterpret_cast<const uint8_t*>(final_hash);
    if ((h[0] == 0x00) && (h[1] == 0x00) && ((h[2] & 0xF0) == 0x00)) {
            //printf("some partial sol %lx\n", seed_n);
            // OPTIMIZATION: Vectorized nonce extraction (direct 64-bit load)
            const uint64_t* nonce_ptr = reinterpret_cast<const uint64_t*>(seed + 232);
            *d_found_nonce = *nonce_ptr;

            const uint32_t* p228 = reinterpret_cast<const uint32_t*>(seed + 228);
            *d_found_u32_at_228 = *p228;


            unsigned long long idx = atomicAdd(ring_tail, 1ULL);
            int pos = (int)(idx % (unsigned long long)ring_cap);

            // Backpressure / overwrite policy:
            // Easiest robust policy is to wait until slot is empty (flag==0).
            // If you prefer "never block" & allow loss, replace the spin with a try-then-drop.
            while (atomicCAS(&ring_flags[pos], 0, -1) != 0) {
           }

           ring_nonces[pos] = seed_n;
          
           // Make data visible system-wide before flagging FULL.
           __threadfence_system();
          
           // Publish the slot as FULL
           atomicExch(&ring_flags[pos], 1);
    }
}


// --- helpers (top of file or above the kernel) ---
__device__ __forceinline__ int pack4_sub128(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
    // Pack 4 bytes into signed i8 lanes (A path): (u8 - 128)
    char4 v;
    v.x = (char)((int)b0 - 128);
    v.y = (char)((int)b1 - 128);
    v.z = (char)((int)b2 - 128);
    v.w = (char)((int)b3 - 128);
    return *reinterpret_cast<int*>(&v);
}

__device__ __forceinline__ int pack4_i8(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
    // Pack 4 bytes into signed i8 lanes (B path): (already i8)
    char4 v;
    v.x = (char)b0;
    v.y = (char)b1;
    v.z = (char)b2;
    v.w = (char)b3;
    return *reinterpret_cast<int*>(&v);
}

extern "C" __global__
__launch_bounds__(256, 4)
void solve_nonce_range_fused(
        const uint8_t* __restrict__ d_prefix232, // 232 bytes
        unsigned long long* d_iter_count,
        u64 nonce_start,
        int nonce_count,
        u32* __restrict__ d_hashes,

        // ring
        unsigned long long* __restrict__ ring_nonces, // u64[cap]
        int                 ring_cap,                 // capacity
        int*                ring_flags,               // i32[cap], 0=empty, 1=full
        unsigned long long* __restrict__ ring_tail   // monotonically increasing reservation counter
        )
{
    const int i = threadIdx.y;   // 0..15
    const int j = threadIdx.x;   // 0..15

    // ---- Static small shared (persists across tile iters) ----
    __shared__  __align__(16) uint8_t sh_prefix[232];
    __shared__  __align__(16) uint8_t sh_seed[240];
    __shared__ u32 sh_root[8];
    __shared__ u32 sh_precv[8];
    __shared__ u32 sh_lwords[16];
    __shared__ uint8_t sh_llen; // = 48

    __shared__  __align__(16) int32_t tileC[16 * 16];

    // Copy common 232B prefix once per block
    if (i == 0 && j == 0) {
        #pragma unroll
        for (int t = 0; t < 232; ++t) sh_prefix[t] = d_prefix232[t];
    }
    __syncthreads();

    // ---- Dynamic shared ----
    // We will store prepacked A and B (int32 = 4 bytes×4 lanes) here.
    extern __shared__ __align__(16) uint8_t smem[];
    // Layout: [A_packed (16 rows × TILE_K/4 ints)] [PAD] [B_packed (TILE_K/4 groups × 16 cols ints)]
    // Use a small pad to reduce bank conflicts between the two regions.
    int32_t* As4 = reinterpret_cast<int32_t*>(smem);
    int32_t* Bs4 = reinterpret_cast<int32_t*>(smem + (size_t)16 * (TILE_K/4) * sizeof(int32_t) + 64);

    // Process many nonces with a single persistent block
    for (int seed = blockIdx.x; seed < nonce_count; seed += gridDim.x) {

        // Thread (0,0) builds the 240B seed and computes root/preCV/lastWords
        if (i == 0 && j == 0) {
            atomicAdd(d_iter_count, 1ULL);

            #pragma unroll
            for (int t = 0; t < 232; ++t) sh_seed[t] = sh_prefix[t];

            const u64 nonce = nonce_start + (u64)seed;
            store_le64(&sh_seed[232], nonce);

            compute_root_from_seed240(sh_seed, sh_root, sh_precv, sh_lwords, &sh_llen);
        }
        __syncthreads();

        constexpr int K            = 50240;
        constexpr int A_BYTES      = 16 * K;           // 803,840
        constexpr int A_BLOCKS     = A_BYTES / 64;     // 12,560
        constexpr int B_BASE_BLOCK = A_BLOCKS;         // 12,560
        const u32     llen         = (u32)sh_llen;

        int acc   = 0;
        int sum_b = 0;

        const int thread_id     = threadIdx.y * blockDim.x + threadIdx.x; // 0..255
        const int total_threads = blockDim.x * blockDim.y;                 // 256

        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            const int tile    = min(TILE_K, K - k0);
            const int groups  = (tile + 3) >> 2;        // number of 4-byte DP4A groups
            const int strideA = (TILE_K >> 2);          // ints per row
            const int strideB = 16;                     // ints per group (one per column)

            // =========================================================================
            // 1) Produce & PRE-PACK A: (16 rows × tile) → (16 rows × groups) int32
            //    We XOF in 64-byte blocks, each block → 16 ints (each packs 4 bytes).
            // =========================================================================
            const int a_blocks_per_row = (tile + 63) / 64;

            for (int linear_idx = thread_id; linear_idx < 16 * a_blocks_per_row; linear_idx += total_threads) {
                const int ri      = linear_idx / a_blocks_per_row;  // row 0..15
                const int rb      = linear_idx % a_blocks_per_row;  // 64B block index within the row
                const int kk_base = rb * 64;

                if (kk_base < tile) {
                    const uint32_t blkA = (uint32_t)(ri * (K/64) + (k0/64) + rb);

                    u32 words[16];
                    xof_emit_words(blkA, sh_root, sh_precv, sh_lwords, llen, words);
                    const uint8_t* w = reinterpret_cast<const uint8_t*>(words);

                    // 16 packed ints per 64B block
                    #pragma unroll
                    for (int q = 0; q < 16; ++q) {
                        const int gidx = (kk_base >> 2) + q;   // which 4-byte group along k
                        if (gidx < groups) {
                            // Handle tail: pad past tile with zeros
                            const int byte0_k = kk_base + 4*q + 0;
                            const int byte1_k = kk_base + 4*q + 1;
                            const int byte2_k = kk_base + 4*q + 2;
                            const int byte3_k = kk_base + 4*q + 3;

                            uint8_t b0 = (byte0_k < tile) ? w[4*q + 0] : 0;
                            uint8_t b1 = (byte1_k < tile) ? w[4*q + 1] : 0;
                            uint8_t b2 = (byte2_k < tile) ? w[4*q + 2] : 0;
                            uint8_t b3 = (byte3_k < tile) ? w[4*q + 3] : 0;

                            As4[ri * strideA + gidx] = pack4_sub128(b0, b1, b2, b3);
                        }
                    }
                }
            }

            // =========================================================================
            // 2) Produce & PRE-PACK B: (tile × 16) but store as groups along k for each col j
            //    For each group (kk..kk+3), pack B[kk..kk+3][j] into one int32.
            // =========================================================================
            for (int gb = thread_id; gb < groups; gb += total_threads) {
                const int kk_base = gb * 4;
                const uint32_t blkB = (uint32_t)(B_BASE_BLOCK + ((k0 + kk_base) >> 2));

                u32 words[16];
                xof_emit_words(blkB, sh_root, sh_precv, sh_lwords, llen, words);
                const uint8_t* w = reinterpret_cast<const uint8_t*>(words);
                // w layout: 4 consecutive 16-byte “columns” for kk, kk+1, kk+2, kk+3:
                // [ j0..j15 | j0..j15 | j0..j15 | j0..j15 ]

                #pragma unroll
                for (int jj = 0; jj < 16; ++jj) {
                    const int k0_ok = (kk_base + 0) < tile;
                    const int k1_ok = (kk_base + 1) < tile;
                    const int k2_ok = (kk_base + 2) < tile;
                    const int k3_ok = (kk_base + 3) < tile;

                    uint8_t b0 = k0_ok ? w[0*16 + jj] : 0;
                    uint8_t b1 = k1_ok ? w[1*16 + jj] : 0;
                    uint8_t b2 = k2_ok ? w[2*16 + jj] : 0;
                    uint8_t b3 = k3_ok ? w[3*16 + jj] : 0;

                    Bs4[gb * strideB + jj] = pack4_i8(b0, b1, b2, b3);
                }
            }

            __syncthreads();

            // =========================================================================
            // 3) Compute:  One DP4A per 4-k group. Also sum_b via DP4A with 0x01010101.
            // =========================================================================
            const int ONES = 0x01010101;
            for (int g = 0; g < groups; ++g) {
                const int a_p = As4[i * strideA + g];
                const int b_p = Bs4[g * strideB + j];

                acc   = __dp4a(a_p, b_p, acc);         // 4 MACs
                sum_b = __dp4a(ONES, b_p, sum_b);      // sum of 4 int8s in one op
            }

            __syncthreads();
        }

        // Undo A bias: (A_u8 - 128) * b  =>  add back 128 * sum_b
        acc += (sum_b << 7);

        tileC[i * 16 + j] = acc;

        __syncthreads();
        if (i == 0 && j == 0 && seed == 0) {
            for (int g = 0; g < 64; ++g) {
              d_hashes[g] = (u32)tileC[g];
            }
        }
        u64 d_found_nonce;
        u32 d_found_u32_at_228;

        fused_blake3_hash_and_detect(
                sh_seed, 
                tileC,
                &d_found_nonce,
                &d_found_u32_at_228,
                nonce_start + seed,

        ring_nonces, // u64[cap]
        ring_cap,                 // capacity
        ring_flags,               // i32[cap], 0=empty, 1=full
        ring_tail   // monotonically increasing reservation counter

                );
    }
}


