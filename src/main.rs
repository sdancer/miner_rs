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

//use cudarc::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg};
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::sys::cuDeviceGet;
use cudarc::driver::sys::cuDeviceGetAttribute;
use cudarc::driver::sys::{cuDeviceGetCount, cuInit};
use std::collections::HashMap;
use std::ffi::c_uint;
use std::sync::Arc;
// ... keep all your existing imports, helpers, and functions ...
fn get_device_cc(ordinal: i32) -> (i32, i32) {
    unsafe {
        let mut dev = 0;
        let _ = cuDeviceGet(&mut dev as *mut _, ordinal);
        let mut major = 0;
        let mut minor = 0;
        let _ = cuDeviceGetAttribute(
            &mut major as *mut _,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            dev,
        );
        let _ = cuDeviceGetAttribute(
            &mut minor as *mut _,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            dev,
        );
        (major, minor)
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---------- host precompute ----------
    let seed = [0u8; 240];

    let start = std::time::Instant::now();
    // (Optional) keep your CPU matmul + hash preview
    {
        let x = cpu_ref::calculate_matmul(&seed); // Vec<i32>
        println!("matmul seed 0 (first 32 i32): {:?}", &x[..32]);

        // seed || matmul bytes → BLAKE3
        let mut buf = Vec::with_capacity(seed.len() + x.len() * 4);
        buf.extend_from_slice(&seed);
        for v in &x {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let h = blake3::hash(&buf);
        println!("seed||matmul BLAKE3 = {}", h.to_hex());
    }

    // ---------- compile PTX once ----------
    unsafe { cuInit(0) };
    let mut dev_count_i32: i32 = 0;
    unsafe {
        let _ = cuDeviceGetCount(&mut dev_count_i32 as *mut i32);
    }
    let dev_count = (dev_count_i32.max(0)) as usize;
    println!("Found {} CUDA device(s).", dev_count);

    let mut archs: Vec<String> = Vec::new();
    for i in 0..dev_count_i32 {
        let (maj, min) = get_device_cc(i);
        let arch = format!("compute_{}{}", maj, min);
        println!("[GPU {}] CC {}.{} -> {}", i, maj, min, arch);
        archs.push(arch);
    }

    // compile PTX once per unique arch
    let mut ptx_by_arch: HashMap<String, cudarc::nvrtc::Ptx> = HashMap::new();
    for arch in archs
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
    {
        // Make a 'static str for CompileOptions.arch
        let arch_static: &'static str = Box::leak(arch.clone().into_boxed_str());

        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(arch_static), // <- &'static str now
            include_paths: vec!["/usr/local/cuda/include".into(), "/opt/cuda/include".into()],
            ..Default::default()
        };

        println!("Compiling PTX for {} ...", arch);
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(PTX_SRC, opts).map_err(|e| {
            eprintln!("NVRTC compile failed for {}: {e}", arch);
            e
        })?;

        ptx_by_arch.insert(arch, ptx);
    }
    println!("PTX compiled for {} arch variant(s).", ptx_by_arch.len()); // ---------- enumerate devices ----------
    let dev_count = (dev_count_i32.max(0)) as usize;
    if dev_count == 0 {
        eprintln!("No CUDA devices found.");
        return Ok(());
    }
    println!("Found {} CUDA device(s).", dev_count);

    // ---------- choose global work to split ----------
    let total_nonce_count: i32 = 1_000_000; // same as your previous `nonce_count`
    let nonce_start_global: u64 = 0;

    // ceil-div to spread remainder
    let per_dev = |n: i32, k: usize| -> i32 { ((n as i64 + k as i64 - 1) / k as i64) as i32 };
    let per = per_dev(total_nonce_count, dev_count);

    // ---------- per-device launch state ----------
    struct DevRun {
        ctx: Arc<CudaContext>,
        d_prefix: cudarc::driver::CudaSlice<u8>,
        d_counter: cudarc::driver::CudaSlice<u64>,
        d_out: cudarc::driver::CudaSlice<i32>,
        out_host: [i32; 256],
        h_counter: [u64; 1],
        nonce_start: u64,
        nonce_count: i32,
        ela: Option<std::time::Duration>,
    }

    let mut runs: Vec<DevRun> = Vec::with_capacity(dev_count);

    // kernel config: you had (16,16,1) and a large grid; keep that unless you change kernel semantics
    let cfg = LaunchConfig {
        block_dim: (16, 16, 1),
        grid_dim: (256 << 6, 1, 1),
        // TILE_K = 256 in kernel → shared = 16*TILE_K + TILE_K*16 = 8192
        shared_mem_bytes: (16 * 256 + 256 * 16) as u32,
    };

    // ---------- setup + launch per device ----------
    for dev_idx in 0..dev_count {
        // per-device nonce slice
        let local_start = nonce_start_global + (dev_idx as u64) * (per as u64);
        // cap final device to not exceed total
        let remaining = total_nonce_count.saturating_sub((dev_idx as i32) * per);
        let local_count = remaining.min(per).max(0);
        if local_count == 0 {
            // no more work
            break;
        }

        let dev_cc = get_device_cc(dev_idx as i32);
        let arch = format!("compute_{}{}", dev_cc.0, dev_cc.1);
        let ptx = ptx_by_arch
            .get(&arch)
            .expect("PTX for arch not found")
            .clone();

        let ctx = CudaContext::new(dev_idx)?; // Arc<CudaContext>
        let stream = ctx.default_stream();

        let module = ctx.load_module(ptx)?;
        let f = module
            .load_function("solve_nonce_range_fused")
            .map_err(|e| {
                eprintln!("[GPU {}] load_function failed: {e}", dev_idx);
                e
            })?;
        // device buffers
        let mut h_counter = [0u64; 1];
        let d_counter = stream.memcpy_stod(&h_counter)?;
        let prefix_host = seed; // 240B; kernel expects [0..232) common + 8B nonce appended
        let d_prefix = stream.memcpy_stod(&prefix_host)?;
        let mut out_host = [0i32; 256];
        let mut d_out = stream.memcpy_stod(&out_host)?;

        // build and launch
        let dev_start = std::time::Instant::now();

        let mut builder = stream.launch_builder(&f);
        // solve_nonce_range_fused(d_prefix232, d_counter, nonce_start, nonce_count, d_C)
        builder.arg(&d_prefix);
        builder.arg(&d_counter);
        builder.arg(&local_start);
        builder.arg(&local_count);
        builder.arg(&mut d_out);

        unsafe { builder.launch(cfg) }?;
        stream.synchronize()?;

        let ela = dev_start.elapsed();

        runs.push(DevRun {
            ctx,
            d_prefix,
            d_counter,
            d_out,
            out_host,
            h_counter,
            nonce_start: local_start,
            nonce_count: local_count,
            ela: Some(ela),
        });
    }

    // ---------- gather results ----------
    let mut total_iters: u64 = 0;
    for (i, run) in runs.iter_mut().enumerate() {
        let stream = run.ctx.default_stream();
        stream.memcpy_dtoh(&run.d_out, &mut run.out_host)?;
        stream.memcpy_dtoh(&run.d_counter, &mut run.h_counter)?;

        total_iters += run.h_counter[0];

        // pretty print small preview to avoid spam
        let bytes = map_to_binary_host(&run.out_host);

        println!(
            "\n[GPU {}] nonce_start={}, nonce_count={}, elapsed={:?}",
            i,
            run.nonce_start,
            run.nonce_count,
            run.ela.unwrap()
        );
        print_tensor_bytes_grid_head(&bytes, 2);
        println!("[GPU {}] iterations = {}", i, run.h_counter[0]);
    }

    println!("\nAll devices done. Total iterations = {}", total_iters);
    Ok(())
}

// Same as yours, but this variant prints only `rows_head` rows (defaults to 2 above)
fn print_tensor_bytes_grid_head(bytes: &[u8], rows_head: usize) {
    assert_eq!(bytes.len(), 1024);
    println!("tensor_c (bytes; 16 rows × 64 bytes, groups = 1 i32):");
    for r in 0..rows_head {
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
