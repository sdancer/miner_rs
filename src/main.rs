use cudarc::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

const PTX_SRC: &str = include_str!("miner.cu");

use core::mem;

use blake3::platform::{Platform};

/// Tiny hex (lowercase) helper without pulling a crate.
fn hex_lower(data: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(data.len() * 2);
    for &b in data {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

const BLOCK_LEN : usize = 64;

use arrayref::{array_mut_ref, array_ref};

#[inline(always)]
pub fn le_bytes_from_words_32(words: &[u32; 8]) -> [u8; 32] {
    let mut out = [0; 32];
    *array_mut_ref!(out, 0 * 4, 4) = words[0].to_le_bytes();
    *array_mut_ref!(out, 1 * 4, 4) = words[1].to_le_bytes();
    *array_mut_ref!(out, 2 * 4, 4) = words[2].to_le_bytes();
    *array_mut_ref!(out, 3 * 4, 4) = words[3].to_le_bytes();
    *array_mut_ref!(out, 4 * 4, 4) = words[4].to_le_bytes();
    *array_mut_ref!(out, 5 * 4, 4) = words[5].to_le_bytes();
    *array_mut_ref!(out, 6 * 4, 4) = words[6].to_le_bytes();
    *array_mut_ref!(out, 7 * 4, 4) = words[7].to_le_bytes();
    out
}

type CVWords = [u32; 8];

/// Compare folded CV (8 u32) against GPU result that also folded (if you do).
pub fn test_cpu_cv_vs_gpu_zero() {
    let plat = Platform::detect();

    // chaining value: 8×u32 all zeros
    let cv : CVWords = [0; 8];

    // 64-byte block all zeros
    let block: [u8; BLOCK_LEN] = [0u8; BLOCK_LEN];

    // match GPU params
    let counter: u64 = 0;
    let flags: u8 = 0;
    let block_len: u8 = 64;

    let cv_bytes = plat.compress_xof(&cv, &block, block_len, counter, flags);

    // Print as bytes (LE) for easy GPU-side comparison
    println!("CPU folded CV (32 bytes): {}", hex_lower(&cv_bytes));

}

fn main() -> Result<(), DriverError> {
    let start = std::time::Instant::now();

    test_cpu_cv_vs_gpu_zero();

    let ptx = compile_ptx(PTX_SRC).unwrap();
    println!("Compilation succeeded in {:?}", start.elapsed());

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    println!("Built in {:?}", start.elapsed());

    let module = ctx.load_module(ptx)?;
    let f = module.load_function("compress")?;
    println!("Loaded in {:?}", start.elapsed());

    let a_host = [0u8; 240];
    let b_host = [0u8; 240];
    let mut c_host = [0u8; 64];

    let chaining_value = stream.memcpy_stod(&a_host)?;
    let block_words = stream.memcpy_stod(&b_host)?;
    let mut state_out = stream.memcpy_stod(&c_host)?;

    println!("Copied in {:?}", start.elapsed());

    let mut builder = stream.launch_builder(&f);
//const u32 *__restrict__ chaining_value,  // cv[8]
//    const u32 *__restrict__ block_words,     // m[16]
//    u64 counter,
//    u32 block_len,
//    u32 flags,
//    u32 *__restrict__ state_out)             // writes v[16]
//{
    builder.arg(&chaining_value);
    builder.arg(&block_words);
    builder.arg(&0);
    builder.arg(&64);
    builder.arg(&0);
    builder.arg(&mut state_out);

    let cfg = LaunchConfig {
        block_dim: (1, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { builder.launch(cfg) }?;

    stream.memcpy_dtoh(&state_out, &mut c_host)?;
    println!("Found {:?} in {:?}", hex_lower(&c_host), start.elapsed());
    Ok(())
}

