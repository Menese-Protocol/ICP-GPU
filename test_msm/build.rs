use std::process::Command;
use std::path::Path;

fn main() {
    let cuda_src = "../cuda/msm_api.cu";
    let cuda_obj = "../cuda/msm_api.o";

    // Only recompile if source changed
    println!("cargo:rerun-if-changed={}", cuda_src);

    if !Path::new(cuda_obj).exists() {
        let status = Command::new("nvcc")
            .args([
                "-O2", "-arch=sm_120",
                "-I/workspace/sppark",
                "-I/workspace/sppark/ff",
                "-I/workspace/sppark/ec",
                "-I/workspace/sppark/msm",
                "-I/workspace/sppark/util",
                "-I/workspace/blst/src",
                "--expt-relaxed-constexpr",
                "-DFEATURE_BLS12_381",
                "-c", cuda_src,
                "-o", cuda_obj,
            ])
            .status()
            .expect("Failed to run nvcc");
        assert!(status.success(), "nvcc compilation failed");
    }

    // Link the CUDA object
    let cuda_dir = std::fs::canonicalize("../cuda").unwrap();
    println!("cargo:rustc-link-search=native={}", cuda_dir.display());
    println!("cargo:rustc-link-lib=static=msm_api");

    // Link blst for host-side field arithmetic
    println!("cargo:rustc-link-search=native=/workspace/blst");
    println!("cargo:rustc-link-lib=static=blst");

    // Link CUDA runtime
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
