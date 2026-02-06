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

    // Link with optimized BLAS for GEMM (600+ GFLOPS)
    if (target.result.os.tag == .windows) {
        // Intel MKL 2025.3 on Windows (via winget install Intel.oneMKL)
        // mkl_rt.dll handles all threading automatically (uses TBB)
        const mkl_inc = b.fmt("{s}/include", .{mkl_root});
        const mkl_lib = b.fmt("{s}/lib", .{mkl_root});
        lib.addIncludePath(.{ .cwd_relative = mkl_inc });
        lib.addLibraryPath(.{ .cwd_relative = mkl_lib });
        lib.linkSystemLibrary("mkl_rt");
    } else {
        // Linux: Dynamic BLAS backend selection at runtime via dlopen
        // Priority: Intel MKL > OpenBLAS-tuned > OpenBLAS system > Zig GEMM fallback
        // Link with dl for dlopen/dlsym (dynamic loading)
        lib.linkSystemLibrary("dl");

        // Also link with system OpenBLAS as fallback (loaded at NIF startup if available)
        lib.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu/openblas-pthread/" });
        lib.addIncludePath(.{ .cwd_relative = "/usr/include/x86_64-linux-gnu/openblas-pthread/" });
        lib.linkSystemLibrary("openblas");
    }

    // Install the library
    b.installArtifact(lib);
}
