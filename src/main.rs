use cudarc::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

const PTX_SRC: &str = include_str!("miner.cu");

fn main() -> Result<(), DriverError> {
    let start = std::time::Instant::now();

    let ptx = compile_ptx(PTX_SRC).unwrap();
    //let program = nvrtc::Program::new("default_program", src);
    //let compile_options = [
    //    "-I/usr/local/cuda/include", // CUDA headers
    //    "-I/usr/include",            // system headers, adjust if needed
    //];
    //program.compile(&compile_options)?;
    println!("Compilation succeeded in {:?}", start.elapsed());

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    println!("Built in {:?}", start.elapsed());

    let module = ctx.load_module(ptx)?;
    let f = module.load_function("compress")?;
    println!("Loaded in {:?}", start.elapsed());

    let a_host = [0u8; 240];
    let b_host = [0u8; 240];
    let mut c_host = [0u8; 32];

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
    builder.arg(&240);
    builder.arg(&0);
    builder.arg(&mut state_out);






    let cfg = LaunchConfig {
        block_dim: (1, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { builder.launch(cfg) }?;

    stream.memcpy_dtoh(&state_out, &mut c_host)?;
    println!("Found {:?} in {:?}", c_host, start.elapsed());
    Ok(())
}

