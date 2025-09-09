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

fn dev_attr(dev: i32, attr: CUdevice_attribute) -> i32 {
    let mut v = 0;
    unsafe {
        cuDeviceGetAttribute(&mut v, attr, dev);
    }
    v
}

// ---------- per-device launch state ----------
struct DevRun {
    ctx: Arc<CudaContext>,
    stream_copy: Arc<cudarc::driver::CudaStream>,
    module: Arc<cudarc::driver::CudaModule>,
    d_prefix: cudarc::driver::CudaSlice<u8>,
    d_counter: cudarc::driver::CudaSlice<u64>,
    d_out: cudarc::driver::CudaSlice<i32>,
    out_host: [i32; 256],
    h_counter: [u64; 1],
    nonce_start: u64,
    nonce_count: i32,
    ela: Option<std::time::Duration>,

    // --- new ring fields ---
    ring_cap: usize,
    d_ring_nonces: cudarc::driver::CudaSlice<u64>, // capacity ring_cap
    d_ring_flags: cudarc::driver::CudaSlice<i32>,  // 0 empty, 1 full
    d_ring_tail: cudarc::driver::CudaSlice<u64>,   // single u64
    d_ring_dropped: cudarc::driver::CudaSlice<u64>, // single u64 (optional)
    h_flags_scratch: Vec<i32>,                     // host scratch to check a window of flags
    h_nonces_scratch: Vec<u64>,                    // host scratch to copy out a window of nonces
    head_host: u64,                                // we keep consumer head only on host
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---------- host precompute ----------
    let seed = [0u8; 240];

    // let start = std::time::Instant::now();
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
    let total_nonce_count: i32 = 0x7fffffff;
    let nonce_start_global: u64 = 0;

    // ceil-div to spread remainder
    // let per_dev = |n: i32, k: usize| -> i32 { ((n as i64 + k as i64 - 1) / k as i64) as i32 };
    // let per = per_dev(total_nonce_count, dev_count);

    let mut runs: Vec<DevRun> = Vec::with_capacity(dev_count);

    // ---------- setup + launch per device ----------
    for dev_idx in 0..dev_count {
        let max_threads = dev_attr(
            dev_idx as i32,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        );
        println!("[GPU {}] maxThreadsPerBlock={}", dev_idx, max_threads);
        println!(
            "[GPU {}] maxGridDimX={}",
            dev_idx,
            dev_attr(
                dev_idx as i32,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X
            )
        );
        println!(
            "[GPU {}] maxSharedMemPerBlockOptin={} B",
            dev_idx,
            dev_attr(
                dev_idx as i32,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
            )
        );
        let threads: u32 = if 256 << 6 > max_threads {
            max_threads.try_into().unwrap()
        } else {
            256 << 6
        };
        let cfg = LaunchConfig {
            block_dim: (16, 16, 1),
            grid_dim: (threads, 1, 1),
            // TILE_K = 256 in kernel → shared = 16*TILE_K + TILE_K*16 = 8192
            shared_mem_bytes: (16 * 256 + 256 * 16) as u32,
        };

        // per-device nonce slice
        // let local_start = nonce_start_global + (dev_idx as u64) * (per as u64);
        // cap final device to not exceed total
        // let remaining = total_nonce_count.saturating_sub((dev_idx as i32) * per);
        // let local_count = remaining.min(per).max(0);
        // if local_count == 0 {
        //     // no more work
        //     break;
        // }
        //
        let local_start: u64 = <usize as TryInto<u64>>::try_into(dev_idx).unwrap() * 0x1_00000000;

        let local_count = 0x7fffffff;

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
        println!("{} loaded ", dev_idx);
        // device buffers
        let h_counter = [0u64; 1];
        let d_counter = stream.memcpy_stod(&h_counter)?;
        let prefix_host = seed; // 240B; kernel expects [0..232) common + 8B nonce appended
        let d_prefix = stream.memcpy_stod(&prefix_host)?;
        let out_host = [0i32; 256];
        let mut d_out = stream.memcpy_stod(&out_host)?;

        // build and launch
        let dev_start = std::time::Instant::now();

        let ring_cap: usize = 4096; // e.g., 65536 slots; tune as you like (solutions are rare)
        let zero_u64 = [0u64; 1];

        // Device allocations
        let d_ring_nonces = stream.alloc_zeros::<u64>(ring_cap)?;
        let d_ring_flags = stream.alloc_zeros::<i32>(ring_cap)?;
        let d_ring_tail = stream.memcpy_stod(&zero_u64)?; // start at 0
        let d_ring_dropped = stream.memcpy_stod(&zero_u64)?; // optional

        // Host scratch
        let h_flags_scratch = vec![0i32; 4096]; // small probe window
        let h_nonces_scratch = vec![0u64; 4096];

        let mut builder = stream.launch_builder(&f);
        // solve_nonce_range_fused(d_prefix232, d_counter, nonce_start, nonce_count, d_C)
        builder.arg(&d_prefix);
        builder.arg(&d_counter);
        builder.arg(&local_start);
        builder.arg(&local_count);
        builder.arg(&mut d_out);

        builder.arg(&d_ring_nonces);
        let ring_cap_i32 = ring_cap as i32;
        builder.arg(&ring_cap_i32);
        builder.arg(&d_ring_flags);
        builder.arg(&d_ring_tail);

        println!("{} launching ", dev_idx);

        unsafe { builder.launch(cfg) }?;

        println!("{} launched ", dev_idx);

        // stream.synchronize()?;

        let ela = dev_start.elapsed();

        println!("{} ela: launched ", dev_idx);

       let stream_copy = ctx.new_stream()?; // separate copy stream
 
std::mem::forget(f);

        runs.push(DevRun {
            ctx,
stream_copy,
            module,
            d_prefix,
            d_counter,
            d_out,
            out_host,
            h_counter,
            nonce_start: local_start,
            nonce_count: local_count,
            ela: Some(ela),

            ring_cap,
            d_ring_nonces,
            d_ring_flags,
            d_ring_tail,
            d_ring_dropped,
            h_flags_scratch,
            h_nonces_scratch,
            head_host: 0,
        });
    }

    // ---------- gather results ----------
    println!("gathering results (non-blocking drain)");

    // Example: poll for ~5 seconds or until you decide to stop.
    let t0 = std::time::Instant::now();
    let poll_for = std::time::Duration::from_secs(5);

    let mut total_iters: u64 = 0;
    let mut all_solutions: Vec<(usize, u64)> = Vec::new();

    'outer: loop {
        println!("tick ");

        for (i, run) in runs.iter_mut().enumerate() {
            // Drain any available solutions from this device's ring
            let sols = drain_ring_once(run)?;
            for nonce in sols {
                all_solutions.push((i, nonce));
            }

            // Optionally also check the iteration counter occasionally
            let stream = run.ctx.default_stream();
            stream.memcpy_dtoh(&run.d_counter, &mut run.h_counter).ok();
            // Don’t spam; just maintain an aggregate
            total_iters = run.h_counter[0];
        }

        // Print any new solutions we got this tick
        if !all_solutions.is_empty() {
            for (gpu, nonce) in all_solutions.drain(..) {
                println!("[GPU {}] SOLUTION nonce=0x{:016x}", gpu, nonce);
            }
        }

        if t0.elapsed() > poll_for {
            break 'outer;
        }

        // light sleep to avoid hogging CPU
        std::thread::sleep(std::time::Duration::from_millis(100));
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

fn drain_ring_once(run: &mut DevRun) -> anyhow::Result<Vec<u64>> {
    let stream = &run.stream_copy;
    let cap = run.ring_cap as usize;

    println!("got stream");
 
    // 1) Pull the entire flags array to host
    let mut h_flags = vec![0i32; cap];
    stream
        .memcpy_dtoh(&run.d_ring_flags, &mut h_flags)
        .map_err(|e| anyhow::anyhow!("memcpy flags D2H failed: {e}"))?;

    println!("copied");
 
    let mut collected = Vec::new();
    let zero = [0i32]; // reusable 1-element zero slice
    let mut one_nonce = [0u64]; // reusable 1-element nonce buffer

    //stream.synchronize()?;

    // 2) Scan flags; for each == 1, copy the nonce and clear the flag on device
    for i in 0..cap {
        if h_flags[i] == 1 {
            println!("h_flags[{i}]");
            // 2a) Read nonce i
            stream
                .memcpy_dtoh(&run.d_ring_nonces.slice(i..i + 1), &mut one_nonce)
                .map_err(|e| anyhow::anyhow!("memcpy nonce[{i}] D2H failed: {e}"))?;

            collected.push(one_nonce[0]);

            // 2b) Clear flag[i] back on device
            let mut d_flag_i: cudarc::driver::CudaViewMut<i32> = run
                .d_ring_flags
                .try_slice_mut(i..i + 1)
                .ok_or_else(|| anyhow::anyhow!("flag slice OOB at {i}"))?;

            stream
                .memcpy_htod(&zero, &mut d_flag_i)
                .map_err(|e| anyhow::anyhow!("memcpy flag[{i}] H2D failed: {e}"))?;
        }
    }

    // (Optional) keep a simple head accounting if you want it
    // run.head_host = (run.head_host + collected.len() as u64) % run.ring_cap as u64;

    // Ensure all H2D clears are done before returning
    // stream.synchronize()?;

    Ok(collected)
}

/*
fn drain_ring_once(run: &mut DevRun) -> anyhow::Result<Vec<u64>> {
    // We’ll inspect up to a window ahead of head_host
    let cap = run.ring_cap as u64;
    let head = run.head_host;
    let window = run.h_flags_scratch.len() as u64;

    // Map ring to a linear window [head .. head+window), wrapping mod cap.
    // Handle wrap in two segments max.

    let mut collected = Vec::new();
    let stream = run.ctx.default_stream();

    let mut remain = window.min(cap); // never look more than cap
    let mut cursor = head;
    while remain > 0 {
        let seg_pos = (cursor % cap) as usize;
        let seg_len = remain.min(cap - (cursor % cap)) as usize;

        let mut h_flags_scratch = vec![0i32; 4096];

        // 1) Copy flags segment
        stream.memcpy_dtoh(
            &run.d_ring_flags,
            &mut h_flags_scratch,
        ).
           map_err(|e| {
                eprintln!("[GPU ] memcpy_dtoh failed: {seg_pos} {seg_len} {e}");
                e
            })?;


        // 2) Scan flags; for each FULL slot, copy corresponding nonce(s)
        let mut block_start = None::<usize>;
        for i in 0..seg_len {
            if h_flags_scratch[i] == 1 {
                if block_start.is_none() {
                    block_start = Some(i);
                }
            } else if let Some(bs) = block_start.take() {
                // flush block [bs..i)
                let n = i - bs;
                // copy n nonces in one go
                stream.memcpy_dtoh(
                    &run.d_ring_nonces.slice(seg_pos + bs..seg_pos + i),
                    &mut run.h_nonces_scratch[..n],
                )?;
                // reset flags back to 0 (free slots)

                // when clearing [seg_pos + bs .. seg_pos + i) with length n
                for j in 0..n {
                    let mut one: cudarc::driver::CudaViewMut<i32> = run
                        .d_ring_flags
                        .try_slice_mut(seg_pos + bs + j..seg_pos + bs + j + 1)
                        .ok_or_else(|| anyhow::anyhow!("oob"))?;
                    stream.memcpy_htod(&[0i32], &mut one)?;
                }

                // append to collected
                collected.extend_from_slice(&run.h_nonces_scratch[..n]);

                // advance head by n
                run.head_host += n as u64;
                cursor += n as u64;
            }
        }

        // tail case: if a block ends at seg end
        if let Some(bs) = block_start {
            let n = seg_len - bs;
            stream.memcpy_dtoh(
                &run.d_ring_nonces.slice(seg_pos + bs..seg_pos + seg_len),
                &mut run.h_nonces_scratch[..n],
            )?;

            let mut one: cudarc::driver::CudaViewMut<i32> = run
                .d_ring_flags
                .try_slice_mut(seg_pos + bs..seg_pos + seg_len)
                .ok_or_else(|| anyhow::anyhow!("oob"))?;
            stream.memcpy_htod(&vec![0; seg_len], &mut one)?;

            collected.extend_from_slice(&run.h_nonces_scratch[..n]);
            run.head_host += n as u64;
            cursor += n as u64;
        }

        // Move window forward
        let consumed = (cursor - head) as u64;
        if consumed >= window {
            break;
        }
        remain = window - consumed;
    }

    Ok(collected)
}
*/
