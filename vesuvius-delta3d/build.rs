//! Builds the vendored vc-delta3d C library. Mirrors `CMakeLists.txt` + `cmake/modern.cmake`
//! upstream (see `vendor/UPSTREAM.md`); keep the source list in sync when re-vendoring.

const SOURCES: &[&str] = &[
    // CMakeLists.txt
    "delta_filter.c",
    "rans.c",
    "legacy.c",
    "api.c",
    // cmake/modern.cmake
    "indexed_encode.c",
    "indexed_decode.c",
    "ng/ng_native_common.c",
    "ng/ng_native_encode.c",
    "ng/ng_native_decode.c",
    "ng/ng_presets.c",
    "ng/ng_frame_encode.c",
    "ng/ng_frame_decode.c",
    "ng/ng_thresholds.c",
    "ng/ng_screen.c",
    "ng/ng_predict.c",
    "ng/ng_fit.c",
    "ng/ng_fast.c",
    "ng/ng_search.c",
    "ng/ng_medium.c",
    "ng/ng_stream.c",
    "ng/ng_select.c",
    "ng/ng_split.c",
    "ng/ng_slow.c",
    "ng/selector.c",
    "ng/header.c",
];

/// `ng_fixed.c` is compiled once per supported block geometry.
const FIXED_DIMS: &[u32] = &[32, 64, 128, 256];

fn base_build() -> cc::Build {
    let mut build = cc::Build::new();
    build
        .include("vendor/include")
        .include("vendor/src")
        .include("vendor/src/ng")
        .define("VC_DELTA3D_MODERN", "1")
        .opt_level(3)
        .warnings(false);
    if build.get_compiler().is_like_msvc() {
        build.flag("/std:c11");
    } else {
        build.flag("-std=gnu11");
        // Upstream: keeps encoder fits bit-identical across platforms. Harmless for decode.
        build.flag("-ffp-contract=off");
    }
    build
}

fn main() {
    println!("cargo:rerun-if-changed=vendor");

    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let mut build = base_build();
    for src in SOURCES {
        build.file(format!("vendor/src/{src}"));
    }

    // Same source file, different defines: compile each geometry into its own object dir and
    // add the objects to the single archive (the fixed kernels and the rest reference each other).
    for dim in FIXED_DIMS {
        let dir = out_dir.join(format!("ng_fixed{dim}"));
        std::fs::create_dir_all(&dir).unwrap();
        let objects = base_build()
            .file("vendor/src/ng/ng_fixed.c")
            .define("NG_FIXED_GEOMETRY", dim.to_string().as_str())
            .define("NG_BLOCK_DIM", dim.to_string().as_str())
            .out_dir(&dir)
            .compile_intermediates();
        for obj in objects {
            build.object(obj);
        }
    }

    build.compile("vc_delta3d");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        println!("cargo:rustc-link-lib=m");
    }
}
