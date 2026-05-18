fn main() {
    println!("cargo:rerun-if-changed=vendor/c3d.c");
    println!("cargo:rerun-if-changed=vendor/c3d.h");

    let mut build = cc::Build::new();
    build
        .file("vendor/c3d.c")
        .include("vendor")
        .opt_level(3)
        .warnings(false);

    // c3d.c is C23 (uses _Noreturn, etc.). gcc/clang accept it under -std=c2x or
    // -std=c23. Fall back gracefully on compilers that don't know either flag.
    if !build.get_compiler().is_like_msvc() {
        build.flag_if_supported("-std=c23");
        build.flag_if_supported("-std=c2x");
        build.flag_if_supported("-fno-strict-aliasing");
    }

    // Enable host-targeted SIMD on native builds. c3d auto-detects AVX-512 /
    // AVX2 / NEON via predefines and falls back to scalar, so -march=native is
    // the simplest way to pick the best kernel without per-target shims.
    // Skip for cross-compilation (CARGO_CFG_TARGET_ARCH != host arch).
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let host_arch = std::env::var("HOST")
        .ok()
        .and_then(|h| h.split('-').next().map(str::to_string))
        .unwrap_or_default();
    if target_arch == host_arch && !build.get_compiler().is_like_msvc() {
        build.flag_if_supported("-march=native");
    }

    build.compile("c3d");
}
