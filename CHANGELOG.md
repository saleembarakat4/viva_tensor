# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2026-02-06

### Performance Revolution

- **818 GFLOPS** on Windows with Intel MKL (+50% vs PyTorch)
- **649 GFLOPS** on Linux with Intel MKL (+21% vs PyTorch)
- **702 GFLOPS** on CUDA with cuBLAS (RTX 4090)

### Added

- **Multi-backend BLAS**: Auto-detection of Intel MKL, OpenBLAS, CUDA cuBLAS
- **Zig SIMD kernels**: AVX2-vectorized dot, sum, scale, exp, sigmoid, relu, log
- **NIF Resource API**: Zero-copy tensor operations (37 NIF functions)
- **Professional benchmarks**: Statistical rigor with confidence intervals
- **In-place mutations**: `add_mut`, `scale_mut`, `relu_mut` (zero allocation)
- **Fused kernels**: `fused_linear_relu` for efficient inference
- **CUDA backend**: Dynamic cuBLAS loading via dlopen

### Architecture

```
Gleam API → Erlang NIF → Zig SIMD / Intel MKL / CUDA cuBLAS
```

### Benchmarks (Verified)

| Size | viva_tensor | PyTorch | Speedup |
|:----:|:-----------:|:-------:|:-------:|
| 4000×4000 | 592 | 516 | +15% |
| 5000×5000 | 649 | 535 | +21% |

> Methodology: 10 runs, 3 warmup, IQR outlier removal, 95% CI

## [1.3.2] - 2026-01-26

### Fixed
- Removed all unused function arguments (zero warnings build)
- Aligned gleam.toml version with git tags

### Documentation
- Added comprehensive CHANGELOG.md
- Updated README with conv2d/pooling usage examples and diagrams

## [1.3.1] - 2026-01-26

### Performance
- **O(1) array access**: Replaced list traversal with Erlang `:array` for O(1) index access
- **Tail-recursive loops**: Eliminated stack growth in conv2d and pooling
- **Zero intermediate allocations**: Direct index computation without list creation
- Estimated **10-50x speedup** for conv2d on large tensors

### Removed
- NIF stubs (pure Gleam implementation is sufficient)

## [1.3.0] - 2026-01-26

### Added
- **conv2d**: Native 2D convolution supporting multiple input formats
- **pad2d/pad4d**: Zero padding for 2D and 4D tensors
- **max_pool2d**: Max pooling with configurable kernel and stride
- **avg_pool2d**: Average pooling with configurable kernel and stride
- **global_avg_pool2d**: Global average pooling

## [1.2.1] - 2026-01-26

### Added
- **slice**: Tensor slicing with start/end indices

## [1.2.0] - 2026-01-25

### Added
- Quantization support (INT8, NF4, AWQ)
- Auto-backend selection
- 8x memory reduction for quantized tensors

## [1.1.0] - 2026-01-24

### Added
- Named tensors with semantic axes
- Broadcasting operations
- Zero-copy transpose via strides

## [1.0.0] - 2026-01-23

### Added
- Initial release
- Core tensor operations (zeros, ones, fill, from_list)
- Element-wise operations (add, sub, mul, div, scale)
- Reductions (sum, mean, max, min, argmax, argmin)
- Matrix operations (dot, matmul, transpose, outer)
- Shape operations (reshape, flatten, squeeze, unsqueeze)
