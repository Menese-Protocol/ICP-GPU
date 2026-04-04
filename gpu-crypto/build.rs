use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cuda_dir = manifest_dir.join("cuda");

    // Find CUDA toolkit
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = format!("{}/bin/nvcc", cuda_path);

    // Check nvcc exists
    let nvcc_check = Command::new(&nvcc).arg("--version").output();
    if nvcc_check.is_err() || !nvcc_check.unwrap().status.success() {
        // No CUDA — build stub that returns errors
        println!("cargo:warning=CUDA not found, GPU acceleration disabled");
        build_stub(&out_dir);
        return;
    }

    // Detect GPU architecture
    let arch = detect_gpu_arch().unwrap_or_else(|| "sm_80".to_string());

    // Compile CUDA source
    let cuda_src = cuda_dir.join("api.cu");
    let cuda_obj = out_dir.join("gpu_crypto.o");

    let status = Command::new(&nvcc)
        .args(&[
            "-c",
            cuda_src.to_str().unwrap(),
            "-o", cuda_obj.to_str().unwrap(),
            &format!("-arch={}", arch),
            "-O2",
            // Include paths (all relative to cuda/ dir)
            &format!("-I{}", cuda_dir.to_str().unwrap()),
            &format!("-I{}", cuda_dir.join("vendor/sppark").to_str().unwrap()),
            &format!("-I{}", cuda_dir.join("vendor/blst").to_str().unwrap()),
        ])
        .status()
        .expect("Failed to run nvcc");

    if !status.success() {
        panic!("nvcc compilation failed");
    }

    // Create static library
    let lib_path = out_dir.join("libgpu_crypto.a");
    let ar_status = Command::new("ar")
        .args(&["rcs", lib_path.to_str().unwrap(), cuda_obj.to_str().unwrap()])
        .status()
        .expect("Failed to run ar");

    if !ar_status.success() {
        panic!("ar failed to create static library");
    }

    // Tell cargo to link
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=gpu_crypto");
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");

    // Rebuild if CUDA sources change
    println!("cargo:rerun-if-changed=cuda/api.cu");
    println!("cargo:rerun-if-changed=cuda/field_sppark.cuh");
    println!("cargo:rerun-if-changed=cuda/sha256.cuh");
    println!("cargo:rerun-if-changed=cuda/g2_coeffs.h");
}

fn detect_gpu_arch() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(&["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;

    if !output.status.success() { return None; }

    let cap = String::from_utf8_lossy(&output.stdout).trim().to_string();
    // Convert "8.6" -> "sm_86", "12.0" -> "sm_120"
    let parts: Vec<&str> = cap.split('.').collect();
    if parts.len() == 2 {
        Some(format!("sm_{}{}", parts[0], parts[1]))
    } else {
        None
    }
}

fn build_stub(out_dir: &PathBuf) {
    // Build a C stub that returns -1 for all GPU functions
    let stub_src = out_dir.join("stub.c");
    std::fs::write(&stub_src, r#"
#include <stdint.h>
int gpu_crypto_init() { return -1; }
int gpu_batch_bls_verify(const uint64_t* a, const uint64_t* b, const uint64_t* c,
                          const uint64_t* d, int32_t* e, int f) { return -1; }
int gpu_batch_sha256(const uint8_t* a, uint8_t* b, int c, int d) { return -1; }
"#).unwrap();

    cc::Build::new()
        .file(&stub_src)
        .compile("gpu_crypto");
}
