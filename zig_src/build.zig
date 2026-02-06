const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Get Erlang include path (needed for nif_entry.c, not for Zig code)
    const erl_include = b.option(
        []const u8,
        "erl_include",
        "Path to Erlang NIF headers",
    ) orelse "/usr/local/lib/erlang/usr/include";

    // Get MKL path for Windows (via winget install Intel.oneMKL)
    const mkl_root = b.option(
        []const u8,
        "mkl_root",
        "Path to Intel MKL installation",
    ) orelse "C:/PROGRA~2/Intel/oneAPI/mkl/latest";

    // Zig 0.15+ API: addLibrary with explicit linkage
    const lib = b.addLibrary(.{
        .name = "viva_tensor_zig",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("viva_zig.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // On Unix, allow undefined symbols (BEAM resolves enif_* at NIF load time)
    // On Windows, ERL_NIF_INIT uses TWinDynNifCallbacks - no undefined symbols
    if (target.result.os.tag != .windows) {
        lib.linker_allow_shlib_undefined = true;
    }

    // Compile the C NIF entry point (uses erl_nif.h for NIF boilerplate)
    lib.addCSourceFile(.{
        .file = b.path("nif_entry.c"),
        .flags = &.{},
    });

    // Add Erlang NIF headers (for nif_entry.c)
    lib.addIncludePath(.{ .cwd_relative = erl_include });

    // Link with libc (needed by nif_entry.c)
    lib.linkLibC();

    // Link with optimized BLAS for GEMM (800+ GFLOPS with MKL)
    // Intel MKL on both Windows and Linux for maximum performance
    if (target.result.os.tag == .windows) {
        // Windows: Intel MKL via winget install Intel.oneMKL
        const mkl_inc = b.fmt("{s}/include", .{mkl_root});
        const mkl_lib = b.fmt("{s}/lib", .{mkl_root});
        lib.addIncludePath(.{ .cwd_relative = mkl_inc });
        lib.addLibraryPath(.{ .cwd_relative = mkl_lib });
        lib.linkSystemLibrary("mkl_rt");
    } else {
        // Linux: Intel MKL (apt install intel-mkl) - 800+ GFLOPS!
        lib.addCSourceFile(.{
            .file = b.path("cuda_gemm.c"),
            .flags = &.{ "-DUSE_MKL_DIRECT" },
        });
        lib.root_module.addCMacro("USE_MKL_DIRECT", "1");

        // MKL headers and libs (Ubuntu: apt install intel-mkl)
        lib.addIncludePath(.{ .cwd_relative = "/usr/include/mkl" });
        lib.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu" });
        lib.linkSystemLibrary("mkl_rt");

        // OpenBLAS as fallback (kept for systems without MKL)
        lib.addLibraryPath(.{ .cwd_relative = "../deps/openblas-tuned/lib" });
        lib.addIncludePath(.{ .cwd_relative = "../deps/openblas-tuned/include" });

        // dlopen for CUDA
        lib.linkSystemLibrary("dl");
        lib.linkSystemLibrary("pthread");
    }

    // Install the library
    b.installArtifact(lib);
}
