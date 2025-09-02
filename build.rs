use std::process::Command;

fn main() {
    // The paths are relative to the project root
    let status = Command::new("nvcc")
        .args([
            "-arch=sm_89",
            "-O3",
            "-Xptxas", "-O3,-v",
            "--use_fast_math",
            "src/miner.cu",
            "-cubin",
            "-o", "miner.cubin",
        ])
        .status()
        .expect("Failed to invoke nvcc");

    if !status.success() {
        panic!("nvcc failed with status {}", status);
    }

    // Tell Cargo to rerun build.rs if miner.cu changes
    println!("cargo:rerun-if-changed=src/miner.cu");
}
