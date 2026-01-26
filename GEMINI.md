# GEMINI.md - viva_tensor

## Project Overview

**viva_tensor** is a pure Gleam tensor library designed as the neural substrate for the **VIVA** (Virtual Intelligent Vida Autônoma) ecosystem. Its primary goal is **"Memory Multiplication"**—turning limited physical VRAM (e.g., 24GB) into massive effective memory (e.g., 96GB+) through mathematical folding and quantization (INT8, NF4, Micro-blocks).

**Philosophy:** "Compression is understanding. Memory is not a bucket, it is a lens."

### Key Features
*   **Pure Gleam:** Core implementation in Gleam with Erlang FFI for specific optimizations (`:array`, `:math`).
*   **Memory Multiplication:** Implements NVFP4-style micro-blocks and INT8 quantization to compress data by 4x-8x.
*   **Zero-Copy Views:** Uses strided tensors (similar to NumPy) to allow O(1) transposes and reshapes without data copying.
*   **Concurrency:** Leverages OTP (Open Telecom Platform) actor pools for parallel tensor operations.
*   **Memory Hierarchy:** Simulates a tiered storage system (GPU -> RAM -> Disk) with automatic offloading.

## Building and Running

The project uses the standard Gleam build toolchain.

*   **Build Project:**
    ```bash
    gleam build
    ```
*   **Run Tests:**
    ```bash
    gleam test
    ```
*   **Format Code:**
    ```bash
    gleam format src test
    ```
*   **Run Benchmarks:**
    ```bash
    gleam run -m viva_tensor/bench           # General benchmarks
    gleam run -m viva_tensor/nf4             # NF4 quantization benchmark
    gleam run -m viva_tensor/flash_attention # Flash attention benchmark
    gleam run -m viva_tensor/compression     # Memory multiplication demo
    ```

## Architecture & Codebase

### Directory Structure
*   `src/viva_tensor.gleam`: Public API entry point. Re-exports core functionality.
*   `src/viva_tensor/tensor.gleam`: Core `Tensor` and `StridedTensor` types, element-wise operations, matmul, and broadcasting logic.
*   `src/viva_tensor/compression.gleam`: Implementation of quantization algorithms (INT8, Q4, NVFP4) and memory hierarchy.
*   `src/viva_tensor/nf4.gleam`: NormalFloat4 quantization (QLoRA style) optimized for neural network weights.
*   `src/viva_tensor/autograd.gleam`: **NEW** Tape-based reverse-mode automatic differentiation engine.
*   `src/viva_tensor/nn.gleam`: **NEW** Neural network layers (Linear) and activation functions.
*   `src/viva_tensor/pool.gleam`: OTP Actor pool implementation for parallelizing tensor operations across BEAM processes.
*   `src/viva_tensor/strided.gleam`: (Implicit in `tensor.gleam`) Logic for handling zero-copy views via strides.
*   `src/viva_tensor_ffi.erl`: Erlang FFI for O(1) array access (`:array`) and mathematical primitives.

### Key Types (`src/viva_tensor/tensor.gleam`)
*   **`Tensor`**: Standard dense tensor backed by a `List(Float)`.
*   **`StridedTensor`**: Advanced tensor backed by an Erlang `:array` with `strides` and `offset`. Enables O(1) slicing and zero-copy manipulation.

### Development Conventions
*   **Functional & Immutable:** All tensor operations return new tensors. Use `StridedTensor` to avoid expensive data copying during reshapes/transposes.
*   **OTP Native:** Heavy computations should potentially utilize the `pool` module to spread load across the BEAM scheduler.
*   **Testing:** Comprehensive tests in `test/` cover constructors, broadcasting, and matrix operations. Always run tests after changes.
*   **FFI:** Use `src/viva_tensor_ffi.erl` sparingly, primarily for performance-critical primitives not efficiently representable in pure Gleam (like O(1) random access).

## Dependencies
*   `gleam_stdlib`: Standard library.
*   `gleamy_bench`: For benchmarking.
*   `gleeunit`: For testing.
