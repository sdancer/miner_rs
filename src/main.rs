use cudarc::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg};

const PTX_SRC: &str = include_str!("miner.cu");

use cudarc::nvrtc::CompileOptions;

use blake3::platform::Platform;
mod cpu_ref;

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

const BLOCK_LEN: usize = 64;

use arrayref::array_mut_ref;

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
    let cv: CVWords = [0; 8];

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
    let seed = [0u8; 240];

    // 1) Just show first 32 elements of the matmul (as before)
    let x = cpu_ref::calculate_matmul(&seed); // Vec<i32>
    println!("matmul seed 0: {:?}", &x[..32]);

    // Serialize i32s to little-endian bytes
    let mut buf = Vec::with_capacity(seed.len() + x.len() * 4);
    buf.extend_from_slice(&seed);
    for v in &x {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    let h = blake3::hash(&buf);
    println!("seed||matmul BLAKE3 = {}", h.to_hex());

    let start = std::time::Instant::now();

    let opts = CompileOptions {
        arch: Some("compute_61"),
        include_paths: vec!["/usr/local/cuda/include".into(), "/opt/cuda/include".into()],
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(PTX_SRC, opts).unwrap();
    println!("Compilation succeeded in {:?}", start.elapsed());

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    println!("Built in {:?}", start.elapsed());

    let module = ctx.load_module(ptx)?;
    let f = module.load_function("solve_nonce_range_fused")?;

    println!("Loaded in {:?}", start.elapsed());


  let mut h_counter = [0u64];
    let mut d_counter = stream.memcpy_stod(&h_counter)?;

    // --- Inputs/outputs ---
    // Prefix is 232 bytes (common), kernel appends 8B LE nonce at [232..239]
    let prefix_host = [0u8; 232];
    let d_prefix = stream.memcpy_stod(&prefix_host)?;

    // Output is a single 16x16 i32 matrix = 256 i32
    let mut out_host = [0i32; 256];
    let mut d_out = stream.memcpy_stod(&out_host)?;

    // Nonce range: start=0, count=1 (single seed)
    let nonce_start: u64 = 0;
    let nonce_count: i32 = 100000;

    println!("Copied in {:?}", start.elapsed());

    // --- Launch config ---
    // Kernel expects block (16,16,1), grid (>=1 blocks). One block = one seed here.
    let cfg = LaunchConfig {
        block_dim: (16, 16, 1),
        grid_dim: (2048, 1, 1),
        // TILE_K = 256 in the kernel → shared = 16*TILE_K + TILE_K*16 bytes = 8192
        shared_mem_bytes: (16 * 256 + 256 * 16) as u32,
    };

    // --- Build args & launch ---
    let mut builder = stream.launch_builder(&f);
    // solve_nonce_range_fused(
    //   const u8* d_prefix232, u64 nonce_start, int nonce_count, int32_t* d_C)
    builder.arg(&d_prefix);
    builder.arg(&d_counter);
    builder.arg(&nonce_start);
    builder.arg(&nonce_count);
    builder.arg(&mut d_out);

    unsafe { builder.launch(cfg) }?;
    stream.synchronize()?;

    // --- Copy back & print 16x16 result ---
    stream.memcpy_dtoh(&d_out, &mut out_host)?;
    let tensor_c_bytes = map_to_binary_host(&out_host);
    print_tensor_bytes_grid(&tensor_c_bytes);
    println!("Done in {:?}", start.elapsed());

stream.memcpy_dtoh(&d_counter, &mut h_counter)?;
println!("iterations = {}", h_counter[0]);
    Ok(())
}

fn map_to_binary_host(c: &[i32; 256]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 * 16 * 4);
    for r in 0..16 {
        for ccol in 0..16 {
            out.extend_from_slice(&c[r * 16 + ccol].to_le_bytes());
        }
    }
    out
}

/// Pretty-print 1024 bytes as 16 rows of 64 bytes (groups of 4).
fn print_tensor_bytes_grid(bytes: &[u8]) {
    assert_eq!(bytes.len(), 1024);
    println!("tensor_c (bytes; 16 rows × 64 bytes, groups = 1 i32):");
    for r in 0..2 {
        let row = &bytes[r * 64..(r + 1) * 64];
        for (k, b) in row.iter().enumerate() {
            if k > 0 && k % 4 == 0 {
                print!(" ");
            }
            print!("{:02x}", b);
        }
        println!();
    }
}
