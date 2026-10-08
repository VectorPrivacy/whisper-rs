#![allow(clippy::uninlined_format_args)]

extern crate bindgen;

use cmake::Config;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

fn main() {
    let target = env::var("TARGET").unwrap();
    // Link C++ standard library
    if let Some(cpp_stdlib) = get_cpp_link_stdlib(&target) {
        println!("cargo:rustc-link-lib=dylib={}", cpp_stdlib);
    }
    // Link macOS Accelerate framework for matrix calculations
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        #[cfg(feature = "coreml")]
        {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=CoreML");
        }
        #[cfg(feature = "metal")]
        {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }

        // whisper.cpp's Metal backend uses @available() checks which compile to
        // ___isPlatformVersionAtLeast calls. Link the Clang compiler runtime to
        // provide this symbol. Use `xcrun clang` to find the system compiler's
        // resource dir (plain `clang` may resolve to an Android NDK toolchain).
        if let Ok(output) = std::process::Command::new("xcrun")
            .args(["clang", "--print-resource-dir"])
            .output()
        {
            if let Ok(resource_dir) = String::from_utf8(output.stdout) {
                let rt_path = format!("{}/lib/darwin", resource_dir.trim());
                println!("cargo:rustc-link-search=native={}", rt_path);
                println!("cargo:rustc-link-lib=static=clang_rt.osx");
            }
        }
    }

    #[cfg(feature = "coreml")]
    println!("cargo:rustc-link-lib=static=whisper.coreml");

    #[cfg(feature = "openblas")]
    {
        if let Ok(openblas_path) = env::var("OPENBLAS_PATH") {
            println!(
                "cargo::rustc-link-search={}",
                PathBuf::from(openblas_path).join("lib").display()
            );
        }
        if cfg!(windows) {
            println!("cargo:rustc-link-lib=libopenblas");
        } else {
            println!("cargo:rustc-link-lib=openblas");
        }
    }
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublasLt");
        println!("cargo:rustc-link-lib=cuda");
        cfg_if::cfg_if! {
            if #[cfg(target_os = "windows")] {
                let cuda_path = PathBuf::from(env::var("CUDA_PATH").unwrap()).join("lib/x64");
                println!("cargo:rustc-link-search={}", cuda_path.display());
            } else {
                println!("cargo:rustc-link-lib=culibos");
                println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
                println!("cargo:rustc-link-search=/usr/local/cuda/lib64/stubs");
                println!("cargo:rustc-link-search=/opt/cuda/lib64");
                println!("cargo:rustc-link-search=/opt/cuda/lib64/stubs");
            }
        }
    }
    #[cfg(feature = "hipblas")]
    {
        println!("cargo:rustc-link-lib=hipblas");
        println!("cargo:rustc-link-lib=rocblas");
        println!("cargo:rustc-link-lib=amdhip64");

        cfg_if::cfg_if! {
            if #[cfg(target_os = "windows")] {
                panic!("Due to a problem with the last revision of the ROCm 5.7 library, it is not possible to compile the library for the windows environment.\nSee https://github.com/ggerganov/whisper.cpp/issues/2202 for more details.")
            } else {
                println!("cargo:rerun-if-env-changed=HIP_PATH");

                let hip_path = match env::var("HIP_PATH") {
                    Ok(path) =>PathBuf::from(path),
                    Err(_) => PathBuf::from("/opt/rocm"),
                };
                let hip_lib_path = hip_path.join("lib");

                println!("cargo:rustc-link-search={}",hip_lib_path.display());
            }
        }
    }

    #[cfg(feature = "openmp")]
    {
        if target.contains("gnu") {
            println!("cargo:rustc-link-lib=gomp");
        } else if target.contains("apple") {
            println!("cargo:rustc-link-lib=omp");
            println!("cargo:rustc-link-search=/opt/homebrew/opt/libomp/lib");
        }
    }

    println!("cargo:rerun-if-changed=wrapper.h");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let whisper_root = out.join("whisper.cpp");

    // WHISPER_PATCHES: a directory of `git diff` patches for the copied sources, applied in name
    // order, so the embedding app carries its whisper.cpp changes without forking whisper.cpp.
    println!("cargo:rerun-if-env-changed=WHISPER_PATCHES");
    let patches = whisper_patches();
    let stamp = patches_stamp(&patches);
    let stamp_path = out.join("whisper.cpp.patches");
    if whisper_root.exists() && std::fs::read_to_string(&stamp_path).ok().as_deref() != Some(stamp.as_str()) {
        std::fs::remove_dir_all(&whisper_root).expect("Failed to remove stale whisper sources");
    }

    if !whisper_root.exists() {
        std::fs::create_dir_all(&whisper_root).unwrap();
        fs_extra::dir::copy("./whisper.cpp", &out, &Default::default()).unwrap_or_else(|e| {
            panic!(
                "Failed to copy whisper sources into {}: {}",
                whisper_root.display(),
                e
            )
        });
        apply_patches(&whisper_root, &patches);
        std::fs::write(&stamp_path, &stamp).expect("Failed to write the patch stamp");
    }

    if env::var("WHISPER_DONT_GENERATE_BINDINGS").is_ok() {
        let _: u64 = std::fs::copy("src/bindings.rs", out.join("bindings.rs"))
            .expect("Failed to copy bindings.rs");
    } else {
        let mut bindings = bindgen::Builder::default().header("wrapper.h");

        #[cfg(feature = "metal")]
        {
            bindings = bindings.header("whisper.cpp/ggml/include/ggml-metal.h");
        }
        #[cfg(feature = "vulkan")]
        {
            bindings = bindings
                .header("whisper.cpp/ggml/include/ggml-vulkan.h")
                .clang_arg("-DGGML_USE_VULKAN=1");
        }

        // bindgen's clang finds no system headers on macOS without the SDK, and the fallback
        // below is generated for another whisper.cpp and platform.
        if target.contains("apple") && env::var_os("SDKROOT").is_none() {
            if let Ok(out) = std::process::Command::new("xcrun").args(["--show-sdk-path"]).output() {
                let sdk = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if out.status.success() && !sdk.is_empty() {
                    bindings = bindings.clang_arg(format!("-isysroot{}", sdk));
                }
            }
        }

        let bindings = bindings
            .clang_arg("-I./whisper.cpp/")
            .clang_arg("-I./whisper.cpp/include")
            .clang_arg("-I./whisper.cpp/ggml/include")
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate();

        match bindings {
            Ok(b) => {
                let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
                b.write_to_file(out_path.join("bindings.rs"))
                    .expect("Couldn't write bindings!");
            }
            Err(e) => {
                println!("cargo:warning=Unable to generate bindings: {}", e);
                println!("cargo:warning=Using bundled bindings.rs, which may be out of date");
                // copy src/bindings.rs to OUT_DIR
                std::fs::copy("src/bindings.rs", out.join("bindings.rs"))
                    .expect("Unable to copy bindings.rs");
            }
        }
    };

    // stop if we're on docs.rs
    if env::var("DOCS_RS").is_ok() {
        return;
    }

    let mut config = Config::new(&whisper_root);

    config
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("WHISPER_ALL_WARNINGS", "OFF")
        .define("WHISPER_ALL_WARNINGS_3RD_PARTY", "OFF")
        .define("WHISPER_BUILD_TESTS", "OFF")
        .define("WHISPER_BUILD_EXAMPLES", "OFF")
        .very_verbose(true)
        .pic(true);

    if cfg!(target_os = "windows") {
        config.cxxflag("/utf-8");
        println!("cargo:rustc-link-lib=advapi32");
    }

    if cfg!(feature = "coreml") {
        config.define("WHISPER_COREML", "ON");
        config.define("WHISPER_COREML_ALLOW_FALLBACK", "1");
    }

    if cfg!(feature = "cuda") {
        config.define("GGML_CUDA", "ON");
        config.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");
        config.define("CMAKE_CUDA_FLAGS", "-Xcompiler=-fPIC");
    }

    if cfg!(feature = "hipblas") {
        config.define("GGML_HIP", "ON");
        config.define("CMAKE_C_COMPILER", "hipcc");
        config.define("CMAKE_CXX_COMPILER", "hipcc");
        println!("cargo:rerun-if-env-changed=AMDGPU_TARGETS");
        if let Ok(gpu_targets) = env::var("AMDGPU_TARGETS") {
            config.define("AMDGPU_TARGETS", gpu_targets);
        }
    }

    if cfg!(feature = "vulkan") {
        config.define("GGML_VULKAN", "ON");
        if target.contains("android") {
            // Android: libvulkan.so is a system library, NDK sysroot provides the linking stub
            println!("cargo:rustc-link-lib=vulkan");
        } else if cfg!(windows) {
            println!("cargo:rerun-if-env-changed=VULKAN_SDK");
            println!("cargo:rustc-link-lib=vulkan-1");
            let vulkan_path = match env::var("VULKAN_SDK") {
                Ok(path) => PathBuf::from(path),
                Err(_) => panic!(
                    "Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set"
                ),
            };
            let vulkan_lib_path = vulkan_path.join("Lib");
            println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
        } else if cfg!(target_os = "macos") {
            println!("cargo:rerun-if-env-changed=VULKAN_SDK");
            println!("cargo:rustc-link-lib=vulkan");
            let vulkan_path = match env::var("VULKAN_SDK") {
                Ok(path) => PathBuf::from(path),
                Err(_) => panic!(
                    "Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set"
                ),
            };
            let vulkan_lib_path = vulkan_path.join("lib");
            println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
        } else {
            println!("cargo:rustc-link-lib=vulkan");
        }
    }

    if cfg!(feature = "openblas") {
        config.define("GGML_BLAS", "ON");
        config.define("GGML_BLAS_VENDOR", "OpenBLAS");
        if env::var("BLAS_INCLUDE_DIRS").is_err() {
            panic!("BLAS_INCLUDE_DIRS environment variable must be set when using OpenBLAS");
        }
        config.define("BLAS_INCLUDE_DIRS", env::var("BLAS_INCLUDE_DIRS").unwrap());
        println!("cargo:rerun-if-env-changed=BLAS_INCLUDE_DIRS");
    }

    if cfg!(feature = "metal") {
        config.define("GGML_METAL", "ON");
        config.define("GGML_METAL_NDEBUG", "ON");
        config.define("GGML_METAL_EMBED_LIBRARY", "ON");
    } else {
        // Metal is enabled by default, so we need to explicitly disable it
        config.define("GGML_METAL", "OFF");
    }

    // Disable BLAS/Accelerate for non-Apple targets (ggml CMakeLists.txt defaults
    // BLAS=ON with Apple vendor when the host is macOS, even during cross-compilation)
    if !target.contains("apple") && !cfg!(feature = "openblas") {
        config.define("GGML_BLAS", "OFF");
        config.define("GGML_ACCELERATE", "OFF");
    }

    // whisper.cpp uses std::filesystem (macOS 10.15+). Ensure the cmake deployment
    // target is at least 10.15 even if the embedding app targets older (e.g. Tauri 10.13).
    if target.contains("apple") {
        let current = env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_default();
        let min_major: u32 = current.split('.').next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let min_minor: u32 = current.split('.').nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        if min_major < 10 || (min_major == 10 && min_minor < 15) {
            config.define("CMAKE_OSX_DEPLOYMENT_TARGET", "10.15");
        }
    }

    if cfg!(debug_assertions) || cfg!(feature = "force-debug") {
        // debug builds are too slow to even remotely be usable,
        // so we build with optimizations even in debug mode
        config.define("CMAKE_BUILD_TYPE", "RelWithDebInfo");
        config.cxxflag("-DWHISPER_DEBUG");
    } else {
        // we're in release mode, explicitly set to release mode
        // see also https://codeberg.org/tazz4843/whisper-rs/issues/226
        config.define("CMAKE_BUILD_TYPE", "Release");
    }

    // Android NDK cross-compilation: cmake 4.x requires the NDK toolchain file
    // (CMAKE_SYSTEM_NAME=Android alone no longer works — cmake injects macOS flags).
    // Map the Rust target triple to the correct ANDROID_ABI so the NDK toolchain
    // compiles for the right architecture.
    if target.contains("android") {
        let ndk = env::var("ANDROID_NDK_HOME")
            .or_else(|_| env::var("ANDROID_NDK"))
            .or_else(|_| env::var("NDK_HOME"))
            .unwrap_or_default();

        if !ndk.is_empty() {
            let toolchain = format!("{}/build/cmake/android.toolchain.cmake", ndk);
            if std::path::Path::new(&toolchain).exists() {
                config.define("CMAKE_TOOLCHAIN_FILE", &toolchain);
            }
        }

        let abi = if target.contains("aarch64") {
            "arm64-v8a"
        } else if target.contains("armv7") {
            "armeabi-v7a"
        } else if target.contains("x86_64") {
            "x86_64"
        } else if target.contains("i686") {
            "x86"
        } else {
            "arm64-v8a"
        };
        config.define("ANDROID_ABI", abi);
        config.define("ANDROID_PLATFORM", "android-26");

        // Vulkan on Android: the NDK only ships C headers (vulkan.h), but ggml-vulkan
        // requires the C++ headers (vulkan.hpp) from Khronos' Vulkan-Headers repo.
        // Also, CMake's find_package(Vulkan) won't find glslc during cross-compilation.
        if cfg!(feature = "vulkan") && !ndk.is_empty() {
            // 1. Fetch Vulkan C++ headers if missing from NDK sysroot
            let out_dir = env::var("OUT_DIR").unwrap();
            let vk_hpp_dir = format!("{}/vulkan-headers", out_dir);
            let vk_hpp_include = format!("{}/include", vk_hpp_dir);
            if !std::path::Path::new(&format!("{}/vulkan/vulkan.hpp", vk_hpp_include)).exists() {
                let status = std::process::Command::new("git")
                    .args(["clone", "--depth", "1",
                           "https://github.com/KhronosGroup/Vulkan-Headers.git",
                           &vk_hpp_dir])
                    .status()
                    .expect("Failed to run git clone for Vulkan-Headers");
                if !status.success() {
                    panic!("Failed to clone Vulkan-Headers (needed for vulkan.hpp on Android)");
                }
            }
            config.define("Vulkan_INCLUDE_DIR", &vk_hpp_include);

            // 2. Point CMake to NDK's bundled glslc for shader compilation
            let host_tag = if cfg!(target_os = "macos") {
                "darwin-x86_64"
            } else if cfg!(windows) {
                "windows-x86_64"
            } else {
                "linux-x86_64"
            };
            let glslc = format!("{}/shader-tools/{}/glslc", ndk, host_tag);
            if std::path::Path::new(&glslc).exists() {
                config.define("Vulkan_GLSLC_EXECUTABLE", &glslc);
            }
        }
    }

    // A release is built on one machine and run on many: -march=native would bake in the build
    // host's instruction set (CI hosts have AVX-512). x86-64-v3 instead: Haswell and Zen 1 onwards.
    // Intel Macs keep x86-64's baseline: Catalina still runs on Ivy Bridge, which lacks AVX2, and
    // they transcribe on Metal anyway.
    if target.starts_with("x86_64") && env::var_os("GGML_NATIVE").is_none() {
        config.define("GGML_NATIVE", "OFF");
        if !target.contains("apple") {
            for isa in ["GGML_SSE42", "GGML_AVX", "GGML_AVX2", "GGML_FMA", "GGML_F16C", "GGML_BMI2"] {
                config.define(isa, "ON");
            }
        }
    }

    // KleidiAI's int8 matmul kernels pick dotprod, i8mm or SME at runtime, where the rest of
    // ggml-cpu is fixed at the cross-compile's baseline (armv8.0 for Android).
    let kleidiai = target.starts_with("aarch64") && (target.contains("android") || target.contains("linux"));
    if kleidiai {
        let src = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("kleidiai");
        config.define("GGML_CPU_KLEIDIAI", "ON");
        config.define("FETCHCONTENT_SOURCE_DIR_KLEIDIAI", &src);
        config.define("FETCHCONTENT_FULLY_DISCONNECTED", "ON");
    }

    // Allow passing any WHISPER, GGML or CMAKE compile flags
    for (key, value) in env::vars() {
        let is_whisper_flag = key.starts_with("WHISPER_")
            && key != "WHISPER_DONT_GENERATE_BINDINGS"
            && key != "WHISPER_PATCHES"
            || key.starts_with("GGML_");
        let is_cmake_flag = key.starts_with("CMAKE_");
        if is_whisper_flag || is_cmake_flag {
            config.define(&key, &value);
        }
    }

    if cfg!(not(feature = "openmp")) {
        config.define("GGML_OPENMP", "OFF");
    }

    if cfg!(feature = "intel-sycl") {
        config.define("BUILD_SHARED_LIBS", "ON");
        config.define("GGML_SYCL", "ON");
        config.define("GGML_SYCL_TARGET", "INTEL");
        config.define("CMAKE_C_COMPILER", "icx");
        config.define("CMAKE_CXX_COMPILER", "icpx");
    }

    let destination = config.build();

    add_link_search_path(&out.join("build")).unwrap();

    println!("cargo:rustc-link-search=native={}", destination.display());
    if cfg!(feature = "intel-sycl") {
        println!("cargo:rustc-link-lib=whisper");
        println!("cargo:rustc-link-lib=ggml");
        println!("cargo:rustc-link-lib=ggml-base");
        println!("cargo:rustc-link-lib=ggml-cpu");
    } else {
        println!("cargo:rustc-link-lib=static=whisper");
        println!("cargo:rustc-link-lib=static=ggml");
        println!("cargo:rustc-link-lib=static=ggml-base");
        println!("cargo:rustc-link-lib=static=ggml-cpu");
    }
    // after ggml-cpu, which calls into it
    if kleidiai {
        println!("cargo:rustc-link-lib=static=kleidiai");
    }
    if target.contains("apple") || cfg!(feature = "openblas") {
        println!("cargo:rustc-link-lib=static=ggml-blas");
    }
    if cfg!(feature = "vulkan") {
        if cfg!(feature = "intel-sycl") {
            println!("cargo:rustc-link-lib=ggml-vulkan");
        } else {
            println!("cargo:rustc-link-lib=static=ggml-vulkan");
        }
    }

    if cfg!(feature = "hipblas") {
        println!("cargo:rustc-link-lib=static=ggml-hip");
    }

    if cfg!(feature = "metal") {
        println!("cargo:rustc-link-lib=static=ggml-metal");
    }

    if cfg!(feature = "cuda") {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
    }

    if cfg!(feature = "openblas") {
        println!("cargo:rustc-link-lib=static=ggml-blas");
    }

    if cfg!(feature = "intel-sycl") {
        println!("cargo:rustc-link-lib=ggml-sycl");
    }

    println!(
        "cargo:WHISPER_CPP_VERSION={}",
        get_whisper_cpp_version(&whisper_root)
            .expect("Failed to read whisper.cpp CMake config")
            .expect("Could not find whisper.cpp version declaration"),
    );

    // for whatever reason this file is generated during build and triggers cargo complaining
    _ = std::fs::remove_file("bindings/javascript/package.json");
}

// From https://github.com/alexcrichton/cc-rs/blob/fba7feded71ee4f63cfe885673ead6d7b4f2f454/src/lib.rs#L2462
fn get_cpp_link_stdlib(target: &str) -> Option<&'static str> {
    if target.contains("msvc") {
        None
    } else if target.contains("apple") || target.contains("freebsd") || target.contains("openbsd") {
        Some("c++")
    } else if target.contains("android") {
        Some("c++_shared")
    } else {
        Some("stdc++")
    }
}

fn add_link_search_path(dir: &std::path::Path) -> std::io::Result<()> {
    if dir.is_dir() {
        println!("cargo:rustc-link-search={}", dir.display());
        for entry in std::fs::read_dir(dir)? {
            add_link_search_path(&entry?.path())?;
        }
    }
    Ok(())
}

fn whisper_patches() -> Vec<PathBuf> {
    let Some(dir) = env::var_os("WHISPER_PATCHES") else {
        println!("cargo:warning=WHISPER_PATCHES is unset: building whisper.cpp unpatched");
        return Vec::new();
    };
    let dir = PathBuf::from(dir);
    println!("cargo:rerun-if-changed={}", dir.display());
    let mut patches: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("WHISPER_PATCHES {}: {}", dir.display(), e))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "patch"))
        .collect();
    patches.sort();
    for p in &patches {
        println!("cargo:rerun-if-changed={}", p.display());
    }
    patches
}

// FNV-1a over whisper.cpp's version and each patch's name and bytes: the copied sources are
// rebuilt when this changes.
fn patches_stamp(patches: &[PathBuf]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    let version = get_whisper_cpp_version(std::path::Path::new("./whisper.cpp")).ok().flatten().unwrap_or_default();
    for b in version.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    for p in patches {
        let name = p.file_name().unwrap().to_string_lossy().into_owned().into_bytes();
        for b in name.into_iter().chain(std::fs::read(p).expect("Failed to read a patch")) {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    format!("{:016x}", h)
}

fn apply_patches(root: &std::path::Path, patches: &[PathBuf]) {
    if patches.is_empty() {
        return;
    }
    // A repository of its own: inside an enclosing one, `git apply` skips paths outside the
    // current directory without an error. The copy's `.git` is the submodule's link, which no
    // longer resolves from here.
    let dot_git = root.join(".git");
    if dot_git.is_dir() {
        std::fs::remove_dir_all(&dot_git).expect("Failed to remove the copied .git");
    } else if dot_git.exists() {
        std::fs::remove_file(&dot_git).expect("Failed to remove the copied .git");
    }
    let git = |args: &[&std::ffi::OsStr]| {
        let status = std::process::Command::new("git")
            .args(args)
            .current_dir(root)
            .status()
            .expect("Failed to run git (needed to apply WHISPER_PATCHES)");
        status.success()
    };
    assert!(git(&["init".as_ref(), "-q".as_ref()]), "git init failed in {}", root.display());
    for p in patches {
        // --ignore-whitespace: a CRLF checkout of whisper.cpp still matches the LF patches
        let ok = git(&["apply".as_ref(), "--whitespace=nowarn".as_ref(), "--ignore-whitespace".as_ref(), p.as_os_str()]);
        assert!(ok, "Failed to apply {}", p.display());
    }
}

fn get_whisper_cpp_version(whisper_root: &std::path::Path) -> std::io::Result<Option<String>> {
    let cmake_lists = BufReader::new(File::open(whisper_root.join("CMakeLists.txt"))?);

    let (mut major, mut minor) = (None, None);
    for line in cmake_lists.lines() {
        let line = line?;

        if let Some(suffix) = line.strip_prefix(r#"project("whisper.cpp" VERSION "#) {
            let whisper_cpp_version = suffix.trim_end_matches(')');
            return Ok(Some(whisper_cpp_version.into()));
        }
        // 1.9 onwards sets the parts separately
        let part = |name: &str| line.strip_prefix(name).map(|v| v.trim_end_matches(')').trim().to_string());
        if let Some(v) = part("set(WHISPER_VERSION_MAJOR ") {
            major = Some(v);
        } else if let Some(v) = part("set(WHISPER_VERSION_MINOR ") {
            minor = Some(v);
        } else if let Some(v) = part("set(WHISPER_VERSION_PATCH ") {
            if let (Some(major), Some(minor)) = (&major, &minor) {
                return Ok(Some(format!("{}.{}.{}", major, minor, v)));
            }
        }
    }

    Ok(None)
}
