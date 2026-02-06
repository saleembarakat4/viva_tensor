//// Configuration Types with Builder Pattern
////
//// Builder pattern: Gang of Four called, they want royalties.
//// But seriously, this beats magic numbers in function calls any day.
////
//// The idea: sensible defaults + fluent customization. You get a working config
//// out of the box, then override only what you need. No more googling
//// "what's the default stride for conv2d in pytorch".
////
//// ## Design Philosophy
////
//// 1. Defaults are production-ready, not toy examples
//// 2. Labelled arguments for required params, builders for optional ones
//// 3. Every config should be printable for debugging (thanks Gleam!)
////
//// ## Example
//// ```gleam
//// import viva_tensor/core/config
////
//// // Start with defaults, customize what matters
//// let cfg = config.conv2d()
////   |> config.with_stride(2)
////   |> config.with_padding(1)
////
//// // Or be explicit about everything
//// let cfg = config.conv2d_new(
////   kernel_h: 5,
////   kernel_w: 5,
////   stride: 2,
////   padding: 1,
//// )
//// ```

// --- Conv2D Configuration ---
//
// The sacred geometry of convolutions. Get these wrong and your model
// either crashes with dimension mismatches or silently produces garbage.
//
// Output size formula (memorize this, it will save your life):
//
//   O = floor((I - K + 2P) / S) + 1
//
// Where:
//   O = output size
//   I = input size
//   K = kernel size
//   P = padding
//   S = stride
//
// With dilation D, effective kernel size becomes: K' = D*(K-1) + 1
// So: O = floor((I - K' + 2P) / S) + 1

/// Conv2D operation configuration.
///
/// Supports all the usual suspects: stride, padding, dilation, groups.
/// Asymmetric values supported (different H and W) because sometimes
/// your input isn't square and you shouldn't have to pretend it is.
pub type Conv2dConfig {
  Conv2dConfig(
    /// Kernel height - typically 1, 3, 5, or 7. 3x3 is the ResNet sweet spot.
    kernel_h: Int,
    /// Kernel width - usually same as height, but asymmetric kernels exist
    kernel_w: Int,
    /// Stride height - controls output spatial reduction. 2 = halve the size.
    stride_h: Int,
    /// Stride width
    stride_w: Int,
    /// Padding height - add zeros around input. padding=kernel/2 keeps size.
    padding_h: Int,
    /// Padding width
    padding_w: Int,
    /// Dilation height - "atrous" convolution, increases receptive field
    /// without adding parameters. DeepLab loves this, value 1 = standard conv.
    dilation_h: Int,
    /// Dilation width
    dilation_w: Int,
    /// Groups for grouped/depthwise convolution. groups=in_channels is depthwise.
    /// MobileNet's secret sauce for efficient inference.
    groups: Int,
  )
}

/// Default Conv2d: 3x3 kernel, stride 1, no padding, no dilation.
///
/// Why 3x3? VGGNet (2014) showed stacking 3x3s beats larger kernels.
/// Two 3x3s have the same receptive field as one 5x5 but fewer params.
/// Three 3x3s = one 7x7 receptive field. This is why ResNet uses 3x3 everywhere.
///
/// Warning: stride=1 + no padding shrinks output by (kernel-1) pixels per side.
/// For "same" output size, use conv2d_same() or add padding=kernel/2.
pub fn conv2d() -> Conv2dConfig {
  Conv2dConfig(
    kernel_h: 3,
    kernel_w: 3,
    stride_h: 1,
    stride_w: 1,
    padding_h: 0,
    padding_w: 0,
    dilation_h: 1,
    dilation_w: 1,
    groups: 1,
  )
}

/// Explicit Conv2d config with labelled arguments.
///
/// Use this when you know exactly what you want and don't need
/// the builder pattern's incremental customization.
pub fn conv2d_new(
  kernel_h kernel_h: Int,
  kernel_w kernel_w: Int,
  stride stride: Int,
  padding padding: Int,
) -> Conv2dConfig {
  Conv2dConfig(
    kernel_h: kernel_h,
    kernel_w: kernel_w,
    stride_h: stride,
    stride_w: stride,
    padding_h: padding,
    padding_w: padding,
    dilation_h: 1,
    dilation_w: 1,
    groups: 1,
  )
}

/// "Same" padding: output spatial size equals input spatial size.
///
/// Computes padding = kernel_size / 2 (integer division).
/// Only works correctly for odd kernel sizes with stride=1.
/// For even kernels or stride>1, you need asymmetric padding (not supported here).
///
/// Note: PyTorch's "same" padding does asymmetric padding. We keep it simple.
pub fn conv2d_same(kernel_h: Int, kernel_w: Int) -> Conv2dConfig {
  Conv2dConfig(
    kernel_h: kernel_h,
    kernel_w: kernel_w,
    stride_h: 1,
    stride_w: 1,
    padding_h: kernel_h / 2,
    padding_w: kernel_w / 2,
    dilation_h: 1,
    dilation_w: 1,
    groups: 1,
  )
}

// --- Conv2d Builder Methods ---
// Fluent API for incremental configuration. Chain with |> for readability.

/// Set uniform stride (same for H and W).
///
/// stride=2 is the standard way to downsample - halves spatial dimensions.
/// More efficient than conv+maxpool and learns the downsampling.
pub fn with_stride(config: Conv2dConfig, stride: Int) -> Conv2dConfig {
  Conv2dConfig(..config, stride_h: stride, stride_w: stride)
}

/// Set separate strides for height and width.
/// Rarely needed, but here for completeness.
pub fn with_stride_hw(
  config: Conv2dConfig,
  stride_h: Int,
  stride_w: Int,
) -> Conv2dConfig {
  Conv2dConfig(..config, stride_h: stride_h, stride_w: stride_w)
}

/// Set uniform padding.
///
/// padding=1 with 3x3 kernel maintains spatial size (stride=1).
/// padding=2 with 5x5 kernel maintains spatial size (stride=1).
/// General rule: padding = (kernel_size - 1) / 2 for "same" output.
pub fn with_padding(config: Conv2dConfig, padding: Int) -> Conv2dConfig {
  Conv2dConfig(..config, padding_h: padding, padding_w: padding)
}

/// Set separate paddings for height and width.
pub fn with_padding_hw(
  config: Conv2dConfig,
  padding_h: Int,
  padding_w: Int,
) -> Conv2dConfig {
  Conv2dConfig(..config, padding_h: padding_h, padding_w: padding_w)
}

/// Set uniform dilation for atrous/dilated convolution.
///
/// dilation=2 means skip every other pixel when applying kernel.
/// Effective kernel size = dilation * (kernel - 1) + 1
/// So 3x3 with dilation=2 has 5x5 receptive field but only 9 params.
///
/// DeepLab uses this for semantic segmentation - large receptive field
/// without the parameter explosion of large kernels.
pub fn with_dilation(config: Conv2dConfig, dilation: Int) -> Conv2dConfig {
  Conv2dConfig(..config, dilation_h: dilation, dilation_w: dilation)
}

/// Set number of groups for grouped convolution.
///
/// groups=1: standard convolution, all inputs connect to all outputs
/// groups=in_channels: depthwise convolution (MobileNet, EfficientNet)
/// groups=N: grouped convolution, splits channels into N independent groups
///
/// Depthwise + 1x1 pointwise = depthwise separable convolution
/// Cuts compute by ~kernel_size^2 with minimal accuracy loss.
pub fn with_groups(config: Conv2dConfig, groups: Int) -> Conv2dConfig {
  Conv2dConfig(..config, groups: groups)
}

/// Set kernel size (height and width).
pub fn with_kernel(
  config: Conv2dConfig,
  kernel_h: Int,
  kernel_w: Int,
) -> Conv2dConfig {
  Conv2dConfig(..config, kernel_h: kernel_h, kernel_w: kernel_w)
}

// --- Pooling Configuration ---
//
// Pooling: the original downsampling before strided convs took over.
// Still useful for global pooling and certain architectures.
//
// Output size formula (same as conv, kernel=pool_size):
//   O = floor((I - pool_size + 2P) / S) + 1

/// Pooling operation configuration.
///
/// MaxPool: take the max in each window. Good for translation invariance.
/// AvgPool: take the mean. Smoother gradients, sometimes better for deep networks.
/// GlobalAvgPool: pool_size = input_size. One value per channel. Classification head.
pub type PoolConfig {
  PoolConfig(
    /// Pool window height
    pool_h: Int,
    /// Pool window width
    pool_w: Int,
    /// Stride height (typically = pool_h for non-overlapping)
    stride_h: Int,
    /// Stride width
    stride_w: Int,
    /// Padding height (usually 0 for pooling)
    padding_h: Int,
    /// Padding width
    padding_w: Int,
  )
}

/// Default pooling: 2x2 window, stride 2, no padding.
///
/// The classic maxpool config: halves spatial dimensions.
/// Non-overlapping windows (stride = pool_size).
///
/// Fun fact: Hinton thinks pooling is a mistake because it throws away
/// spatial information. Capsule networks are his proposed fix.
/// The jury's still out.
pub fn pool() -> PoolConfig {
  PoolConfig(
    pool_h: 2,
    pool_w: 2,
    stride_h: 2,
    stride_w: 2,
    padding_h: 0,
    padding_w: 0,
  )
}

/// Create pool config with explicit size and stride.
pub fn pool_new(pool_size pool_size: Int, stride stride: Int) -> PoolConfig {
  PoolConfig(
    pool_h: pool_size,
    pool_w: pool_size,
    stride_h: stride,
    stride_w: stride,
    padding_h: 0,
    padding_w: 0,
  )
}

/// Set pool window size.
pub fn pool_with_size(
  config: PoolConfig,
  pool_h: Int,
  pool_w: Int,
) -> PoolConfig {
  PoolConfig(..config, pool_h: pool_h, pool_w: pool_w)
}

/// Set pool stride.
pub fn pool_with_stride(config: PoolConfig, stride: Int) -> PoolConfig {
  PoolConfig(..config, stride_h: stride, stride_w: stride)
}

/// Set pool padding.
pub fn pool_with_padding(config: PoolConfig, padding: Int) -> PoolConfig {
  PoolConfig(..config, padding_h: padding, padding_w: padding)
}

// --- Quantization Configuration ---
//
// Where the magic happens for memory-constrained deployment.
// These configs control the accuracy/compression trade-off.

/// NF4 (NormalFloat4) quantization configuration.
///
/// From Dettmers et al. (2023) "QLoRA: Efficient Finetuning of Quantized LLMs"
///
/// Key parameters:
/// - block_size: Number of weights sharing one scale factor. Smaller = more
///   accurate but more overhead. 64 is the sweet spot from the paper.
/// - double_quant: Quantize the scale factors themselves (FP32 -> FP8).
///   Saves 0.37 bits/param with negligible quality loss. Free lunch.
///
/// Memory math for a 7B model:
///   FP16: 7B * 2 bytes = 14GB
///   NF4:  7B * 0.5 bytes + scales = ~3.5GB
///   NF4 + double_quant: ~3.1GB
pub type NF4Config {
  NF4Config(
    /// Block size for quantization. Each block of N weights shares one scale.
    /// Smaller = better quality, more memory overhead.
    /// Paper uses 64, which gives ~0.5 bits overhead per weight.
    block_size: Int,
    /// Double quantization: quantize the FP32 scales to FP8.
    /// Saves 0.37 bits/param with no measurable quality loss.
    /// No reason not to use this.
    double_quant: Bool,
  )
}

/// Default NF4: block_size=64, double_quant=True (paper settings).
///
/// These are the settings from QLoRA that achieved results matching
/// full-precision finetuning. Don't change unless you know why.
pub fn nf4() -> NF4Config {
  NF4Config(block_size: 64, double_quant: True)
}

/// Override NF4 block size.
///
/// Smaller = better quality, more scale overhead.
/// 32: ~0.69 bits/param overhead, slightly better quality
/// 64: ~0.5 bits/param overhead (default, paper setting)
/// 128: ~0.44 bits/param overhead, slightly worse quality
pub fn nf4_with_block_size(config: NF4Config, block_size: Int) -> NF4Config {
  NF4Config(..config, block_size: block_size)
}

/// Enable/disable double quantization.
/// There's really no reason to disable this, but the option exists.
pub fn nf4_with_double_quant(config: NF4Config, enabled: Bool) -> NF4Config {
  NF4Config(..config, double_quant: enabled)
}

/// INT8 quantization configuration.
///
/// The production workhorse. Jacob et al. (2017) showed INT8 inference
/// loses <1% accuracy on ImageNet while being 4x smaller and up to 4x faster.
///
/// Two modes:
/// - Per-tensor (block_size=0): One scale for the whole tensor. Fastest.
/// - Per-block: One scale per block of N elements. More accurate for outliers.
pub type Int8Config {
  Int8Config(
    /// Block size for per-block quantization. 0 = per-tensor (one global scale).
    /// Per-block helps when tensor has outliers in specific regions.
    block_size: Int,
    /// Symmetric quantization: range is [-127, 127], zero maps to zero.
    /// Asymmetric: range is [0, 255] with a zero-point offset.
    /// Symmetric is simpler and faster, asymmetric handles skewed distributions.
    symmetric: Bool,
  )
}

/// Default INT8: per-tensor symmetric quantization.
///
/// Simplest and fastest. Works well when weight distributions are roughly
/// symmetric around zero (which they usually are for trained models).
pub fn int8() -> Int8Config {
  Int8Config(block_size: 0, symmetric: True)
}

/// Set INT8 block size for per-block quantization.
/// 0 = per-tensor quantization (default, fastest).
pub fn int8_with_block_size(config: Int8Config, block_size: Int) -> Int8Config {
  Int8Config(..config, block_size: block_size)
}

/// AWQ (Activation-Aware Weight Quantization) configuration.
///
/// Lin et al. (2023) key insight: not all weights are equal.
/// ~1% of weights cause ~99% of quantization error - the "salient" channels.
/// These correspond to input channels with large activation magnitudes.
///
/// Solution: scale up salient channels before quantization, scale down after.
/// Effectively gives them more bits of precision where it matters.
///
/// Requires calibration data to identify salient channels.
pub type AWQConfig {
  AWQConfig(
    /// Block size for underlying 4-bit quantization
    block_size: Int,
    /// Number of calibration samples to collect activation statistics.
    /// More = better saliency estimation, but diminishing returns after ~128.
    n_calibration: Int,
    /// Scaling factor for salient channels. Higher = more protection for
    /// important weights, but can cause numerical issues if too high.
    /// Paper uses grid search to find optimal value per layer.
    scale_factor: Float,
  )
}

/// Default AWQ: block_size=64, 128 calibration samples, scale=1.0.
///
/// scale_factor=1.0 is a placeholder - in practice you'd tune this
/// per-layer using calibration data. See the paper for the grid search procedure.
pub fn awq() -> AWQConfig {
  AWQConfig(block_size: 64, n_calibration: 128, scale_factor: 1.0)
}

/// Set AWQ block size.
pub fn awq_with_block_size(config: AWQConfig, block_size: Int) -> AWQConfig {
  AWQConfig(..config, block_size: block_size)
}

/// Set number of calibration samples for saliency estimation.
pub fn awq_with_calibration(config: AWQConfig, n: Int) -> AWQConfig {
  AWQConfig(..config, n_calibration: n)
}

// --- Attention Configuration ---
//
// Scaled dot-product attention: the heart of Transformers.
//
// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
//
// That sqrt(d_k) is crucial - without it, dot products grow with dimension
// and softmax saturates to one-hot, killing gradients.
// Vaswani et al. (2017) figured this out and saved us all.

/// Flash Attention configuration.
///
/// Dao et al. (2022) "FlashAttention: Fast and Memory-Efficient Exact Attention"
/// The algorithm that made long-context LLMs practical.
///
/// Key insight: attention is memory-bound, not compute-bound.
/// By tiling the computation to fit in SRAM, we avoid reading/writing
/// the O(n^2) attention matrix to HBM. 2-4x faster, O(n) memory.
///
/// This config is for the attention parameters, not the Flash algorithm itself.
/// The algorithm is handled in the implementation.
pub type AttentionConfig {
  AttentionConfig(
    /// Number of attention heads. More heads = more expressive, more memory.
    /// GPT-3: 96 heads. BERT-base: 12 heads. Choose based on model size.
    num_heads: Int,
    /// Dimension of each head (d_k = d_model / num_heads typically).
    /// 64 is common. Must divide evenly into total embedding dimension.
    head_dim: Int,
    /// Dropout probability on attention weights. 0.0 for inference.
    /// Training typically uses 0.1. Higher for small datasets to prevent overfitting.
    dropout: Float,
    /// Causal (autoregressive) masking. True for decoder-only models (GPT).
    /// Prevents attending to future tokens. False for encoder models (BERT).
    causal: Bool,
    /// Softmax scaling factor. Typically 1/sqrt(head_dim).
    /// The Vaswani et al. (2017) insight that prevents gradient vanishing.
    scale: Float,
  )
}

/// Create attention config with required parameters.
///
/// Scale is automatically set to 1/sqrt(head_dim) per Vaswani et al. (2017).
/// Override with attention_with_scale() if you're doing something exotic
/// (e.g., cosine attention, ALiBi without scaling).
pub fn attention(
  num_heads num_heads: Int,
  head_dim head_dim: Int,
) -> AttentionConfig {
  let scale = 1.0 /. sqrt(int_to_float(head_dim))
  AttentionConfig(
    num_heads: num_heads,
    head_dim: head_dim,
    dropout: 0.0,
    causal: False,
    scale: scale,
  )
}

/// Enable causal (autoregressive) masking.
///
/// For decoder-only models (GPT, LLaMA, etc.) that should only attend
/// to past tokens, not future ones. Implemented as a triangular mask.
pub fn attention_causal(config: AttentionConfig) -> AttentionConfig {
  AttentionConfig(..config, causal: True)
}

/// Set dropout probability on attention weights.
///
/// 0.0 for inference (always).
/// 0.1 is typical for training.
/// Higher (0.2-0.3) for small datasets or aggressive regularization.
pub fn attention_with_dropout(
  config: AttentionConfig,
  dropout: Float,
) -> AttentionConfig {
  AttentionConfig(..config, dropout: dropout)
}

/// Override the softmax scaling factor.
///
/// Default is 1/sqrt(head_dim) which prevents attention logits from growing
/// too large as dimension increases. You might override this for:
/// - Cosine attention (scale=1, use normalized Q and K)
/// - ALiBi without scaling (Press et al., 2022)
/// - Experimental attention variants
pub fn attention_with_scale(
  config: AttentionConfig,
  scale: Float,
) -> AttentionConfig {
  AttentionConfig(..config, scale: scale)
}

// --- Helpers ---
//
// Gleam doesn't have implicit int->float conversion, which is actually
// a good thing for catching bugs. But we need these helpers.

@external(erlang, "math", "sqrt")
fn sqrt(x: Float) -> Float

fn int_to_float(i: Int) -> Float {
  case i >= 0 {
    True -> positive_int_to_float(i, 0.0)
    False -> 0.0 -. positive_int_to_float(0 - i, 0.0)
  }
}

fn positive_int_to_float(i: Int, acc: Float) -> Float {
  case i {
    0 -> acc
    _ -> positive_int_to_float(i - 1, acc +. 1.0)
  }
}
