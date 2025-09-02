
use blake3;
use std::io::Read;

const SOL_SEED_LEN: usize = 240; // fixed in the Elixir reference
//const NONCE_LEN: usize = 12; // also fixed (see UPOW2.tensormath/5)

pub fn calculate_matmul(sol_seed: &[u8]) -> Vec<u8> {
    // --------------------------- constants ---------------------------
    const ROWS: usize = 16;
    const COLS: usize = 16;
    const K: usize = 50_240;
    const MATRIX_A_SZ: usize = ROWS * K; // 803 840 bytes (u8)
    const MATRIX_B_SZ: usize = K * COLS; // 803 840 bytes (i8 / u8 source)
    const DISCARD_SZ: usize = ROWS * 64; // 1 024 bytes (unused stream)

    // --------------------- validate pre-conditions -------------------
    assert_eq!(
        sol_seed.len(),
        SOL_SEED_LEN,
        "sol_seed must be exactly 240 bytes"
    );

    // --------------------------- XOF stream --------------------------
    let mut hasher = blake3::Hasher::new();
    hasher.update(sol_seed);
    let mut xof = hasher.finalize_xof();

    // Read A (u8) -----------------------------------------------------
    let mut a_bytes = vec![0u8; MATRIX_A_SZ];
    xof.read_exact(&mut a_bytes)
        .expect("failed to read matrix A from XOF");

    // Read B (u8), then cast to i8 -----------------------------------
    let mut b_u8 = vec![0u8; MATRIX_B_SZ];
    xof.read_exact(&mut b_u8)
        .expect("failed to read matrix B from XOF");
    let b_i8: Vec<i8> = b_u8.iter().map(|&x| x as i8).collect();

    // Consume the extra 1 024 bytes (B2) so the stream aligns with the BEAM code
    let mut discard = vec![0u8; DISCARD_SZ];
    xof.read_exact(&mut discard)
        .expect("failed to read discard bytes from XOF");

    // ----------------------- matrix multiply -------------------------
    let c_matrix = multiply_matrices(&a_bytes, &b_i8);

    // -------------------------- serialise ----------------------------
    map_to_binary(c_matrix) // Vec<u8> of length 1 024
}

/// Multiplies a 16 × 50 240 `u8` matrix (`a`) by a 50 240 × 16 `i8` matrix (`b`)
/// and returns a 16 × 16 `i32` result.
///
/// * `a.len()` **must** equal `16 * 50_240`  (803 840 bytes)
/// * `b.len()` **must** equal `50_240 * 16`  (803 840 bytes)
///
/// The layout is **row-major** for both inputs, identical to the Elixir code.
pub fn multiply_matrices(a: &[u8], b: &[i8]) -> [[i32; 16]; 16] {
    const ROWS: usize = 16;
    const COLS: usize = 16;
    const K: usize = 50_240;

    // Panic early if the caller gave us the wrong sizes
    assert_eq!(a.len(), ROWS * K);
    assert_eq!(b.len(), K * COLS);

    // C will hold our 16×16 dot-product results
    let mut c = [[0i32; COLS]; ROWS];

    // Classic triple-nested GEMM loop
    for i in 0..ROWS {
        for k in 0..K {
            // Convert once to i32, re-use for all 16 columns
            let a_val = a[i * K + k] as i32;
            let b_row_offset = k * COLS;

            for j in 0..COLS {
                let b_val = b[b_row_offset + j] as i32; // signed!
                c[i][j] += a_val * b_val;
            }
        }
    }

    c
}

/// Serialise a 16×16 matrix of `i32` into a 1 024-byte row-major buffer
/// (little-endian per cell, identical to Elixir’s `map_to_binary/1`).
pub fn map_to_binary(c: [[i32; 16]; 16]) -> Vec<u8> {
    const BYTES_PER_CELL: usize = std::mem::size_of::<i32>(); // 4
    const ROWS: usize = 16;
    const COLS: usize = 16;
    const TOTAL_BYTES: usize = ROWS * COLS * BYTES_PER_CELL; // 1 024

    // Pre-allocate exactly 1 024 bytes so we never reallocate.
    let mut out = Vec::with_capacity(TOTAL_BYTES);

    // Row-major walk, pushing each i32’s LE representation.
    for row in c.iter() {
        for &val in row.iter() {
            out.extend_from_slice(&val.to_le_bytes());
        }
    }

    debug_assert_eq!(out.len(), TOTAL_BYTES);
    out
}

