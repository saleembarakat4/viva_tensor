const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Get Erlang include path from environment or use default
    const erl_include = b.option(
        []const u8,
        "erl_include",
        "Path to Erlang NIF headers",
    ) orelse "/usr/local/lib/erlang/usr/include";

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

    // Allow undefined symbols - BEAM resolves enif_* at NIF load time
    lib.linker_allow_shlib_undefined = true;

    // Add Erlang NIF headers
    lib.root_module.addIncludePath(.{ .cwd_relative = erl_include });

    // Link with libc for erl_nif
    lib.linkLibC();

    // Install the library
    b.installArtifact(lib);
}
