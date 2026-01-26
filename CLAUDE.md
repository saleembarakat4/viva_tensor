# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is viva_tensor

Pure Gleam tensor library with memory multiplication via quantization. Part of the VIVA ecosystem - provides the neural substrate for VIVA's consciousness (memory, embeddings, attention).

**Core Philosophy:** Compression is understanding. Memory is not a bucket, it's a lens. Uses mathematical folding to turn 24GB VRAM into 96GB+ effective memory.

## Build Commands

```bash
# Build
gleam build

# Run tests
gleam test

# Format code
gleam format src test

# Run benchmarks
gleam run -m viva_tensor/bench
gleam run -m viva_tensor/nf4           # NF4 quantization benchmark
gleam run -m viva_tensor/flash_attention # Flash attention benchmark
gleam run -m viva_tensor/compression   # Memory multiplication demo
```

## Architecture

```
viva_tensor/
├── viva_tensor.gleam          # Public API (re-exports from tensor)
├── tensor.gleam               # Core tensor types + operations
├── named.gleam                # Named tensors with semantic axes
├── axis.gleam                 # Axis types (Batch, Seq, Feature, etc.)
├── compression.gleam          # INT8/Q4 quantization + memory hierarchy
├── nf4.gleam                  # NF4 (NormalFloat4) - QLoRA style quantization
├── flash_attention.gleam      # O(n) memory attention algorithm
├── pool.gleam                 # OTP-based parallel tensor operations
├── blackwell.gleam            # Next-gen compression references
└── rtx4090.gleam              # RTX 4090 memory simulation
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `viva_tensor` | Public API - use this for imports |
| `viva_tensor/tensor` | Core Tensor/StridedTensor types, NumPy-style ops |
| `viva_tensor/compression` | INT8/Q4 quantization, memory hierarchy, offloading |
| `viva_tensor/nf4` | NF4 quantization (8x compression, QLoRA-style) |
| `viva_tensor/flash_attention` | Flash Attention O(n) memory algorithm |
| `viva_tensor/pool` | Parallel tensor ops via OTP processes |
| `viva_tensor/named` | Named axes for safer tensor operations |

## Tensor Types

```gleam
// Regular tensor (list-based)
Tensor(data: List(Float), shape: List(Int))

// Strided tensor (Erlang array, O(1) access)
StridedTensor(storage: ErlangArray, shape: List(Int), strides: List(Int), offset: Int)
```

- Use `to_strided()` for O(1) random access
- Use `transpose_strided()` for zero-copy transpose
- StridedTensor shares underlying storage (views)

## Quantization Formats

| Format | Compression | Error | Use Case |
|--------|-------------|-------|----------|
| FP32 | 1x | 0% | Default precision |
| INT8 | 4x | ~0.2% | General inference |
| Q4 | 8x | ~1.3% | GGML-style, uniform |
| NF4 | 8x | ~0.1% | QLoRA-style, gaussian-optimized |

NF4 uses 16 levels from normal distribution quantiles - better for NN weights.

## FFI Layer

Erlang FFI in `src/viva_tensor_ffi.erl`:
- `list_to_array/1` - Convert list to Erlang :array
- `array_get/2` - O(1) array access
- `send_msg/2` - Message passing for pool
- `collect_n/1` - Collect N messages

## Important Patterns

1. **Zero-copy views**: StridedTensor enables transpose/reshape without data copying
2. **OTP parallelism**: pool.gleam spawns BEAM processes (~2KB each) for parallel tensor ops
3. **Memory hierarchy**: GPU → RAM → Disk tiered storage with automatic offloading
4. **Online softmax**: Flash attention uses incremental statistics to avoid n×n materialization

## Testing

```bash
# Run all tests
gleam test

# Tests cover: constructors, element-wise ops, matmul, broadcasting,
# strided tensors, named tensors, axis operations
```
