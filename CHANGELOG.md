# Changelog

All notable changes to this project will be documented in this file.

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
- **conv2d**: Native 2D convolution supporting multiple input formats:
  - Simple: `[H, W] * [KH, KW] -> [H_out, W_out]`
  - Multi-channel: `[C, H, W] * [C, KH, KW] -> [H_out, W_out]`
  - Full batch: `[N, C_in, H, W] * [C_out, C_in, KH, KW] -> [N, C_out, H_out, W_out]`
- **pad2d/pad4d**: Zero padding for 2D and 4D tensors
- **max_pool2d**: Max pooling with configurable kernel and stride
- **avg_pool2d**: Average pooling with configurable kernel and stride
- **global_avg_pool2d**: Global average pooling `[N, C, H, W] -> [N, C, 1, 1]`
- **conv2d_config**: Configuration builder for stride and padding
- **conv2d_same**: Helper for "same" padding (output = input size)

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
