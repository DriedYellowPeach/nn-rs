fn main() {
    // Check if we are building for a specific target
    if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
        // Set the rustc flag for linking to the Accelerate framework on macOS
        println!("cargo:rustc-flags=-l framework=Accelerate");
    }

    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=openblas");
    }
}
