//// FFI - Foreign Function Interface to Erlang
////
//// The escape hatch from pure functional bliss into the world of
//// mutable arrays and hardware-specific optimizations.
////
//// Why we need this:
//// 1. Erlang lists are O(n) for random access. That's death for matrix ops.
//// 2. Erlang's :array gives us O(1) access (technically O(log32 n), close enough).
//// 3. Native NIFs unlock SIMD, BLAS, and GPU backends.
////
//// ## Performance Hierarchy (fastest to slowest)
////
//// 1. Zig SIMD NIF: Hand-tuned SIMD for the hot paths. 10-100x vs pure Gleam.
//// 2. Apple Accelerate NIF: cblas_dgemm on macOS. Ridiculously optimized.
//// 3. Erlang :array: O(1) access, pure Erlang. 10-50x vs lists for matmul.
//// 4. Pure Gleam lists: Beautiful, correct, slow. Fine for small tensors.
////
//// ## Architecture
////
//// We have three acceleration backends that we auto-select from:
//// - viva_tensor_zig: Portable SIMD via Zig. Works everywhere Zig compiles.
//// - viva_tensor_nif: Apple Accelerate on macOS (cblas, vDSP).
//// - viva_tensor_ffi: Pure Erlang fallback. Always works, just slower.
////
//// The ops module auto-selects the best available backend at runtime.

// --- Types ---

/// Erlang :array - the key to O(1) tensor element access.
///
/// Under the hood, it's a tree of 10-element tuples. Technically O(log32 n)
/// but that's effectively O(1) for any reasonable tensor size.
///
/// A 1M element lookup: log32(1_000_000) = ~4 tree traversals.
/// For lists: 500,000 average. That's the difference between
/// matmul taking 1 second vs 2 hours.
///
/// The array is immutable (functional updates create new nodes),
/// but we mostly just read from it during tensor operations.
pub type ErlangArray

// --- Array Operations ---
//
// These wrap Erlang :array operations for O(1) element access.
// Use these instead of list indexing in any hot path.

/// Convert list to Erlang array for O(1) access.
///
/// O(n) to build, but subsequent access is O(1).
/// Worth it for any tensor you'll index more than once.
pub fn list_to_array(lst: List(Float)) -> ErlangArray {
  list_to_array_ffi(lst)
}

/// Get element from array at index - O(1).
///
/// Contrast with list indexing: O(n).
/// For a 1000-element matmul (1000 iterations, each indexing both inputs),
/// that's 2M list traversals vs 2K array lookups. Huge difference.
pub fn array_get(arr: ErlangArray, index: Int) -> Float {
  array_get_ffi(arr, index)
}

/// Get array size - O(1).
pub fn array_size(arr: ErlangArray) -> Int {
  array_size_ffi(arr)
}

/// Convert array back to list - O(n).
///
/// Use this for final output or when you need list operations.
/// Try to stay in array-land as long as possible for hot paths.
pub fn array_to_list(arr: ErlangArray) -> List(Float) {
  array_to_list_ffi(arr)
}

// --- Optimized Array Math ---
//
// Implemented in Erlang for O(1) element access.
// These are the fallback when NIFs aren't available.
// Still 10-50x faster than naive list-based implementations.

/// Dot product using Erlang arrays.
///
/// Performance: ~10-50x faster than list-based for large vectors.
/// The speedup comes entirely from O(1) vs O(n) element access.
pub fn array_dot(a: ErlangArray, b: ErlangArray) -> Float {
  array_dot_ffi(a, b)
}

/// Matrix multiplication using Erlang arrays.
///
/// C[m,n] = A[m,k] @ B[k,n]
///
/// Naive O(mnk) algorithm but with O(1) element access.
/// For 100x100 matrices: ~50x faster than list-based.
///
/// For serious work, use the Zig SIMD or Accelerate NIF backends.
/// This is the reliable fallback that works everywhere.
pub fn array_matmul(
  a: ErlangArray,
  b: ErlangArray,
  m: Int,
  n: Int,
  k: Int,
) -> ErlangArray {
  array_matmul_ffi(a, b, m, n, k)
}

/// Sum all elements - O(n).
pub fn array_sum(arr: ErlangArray) -> Float {
  array_sum_ffi(arr)
}

/// Scale all elements by scalar - O(n).
pub fn array_scale(arr: ErlangArray, scalar: Float) -> ErlangArray {
  array_scale_ffi(arr, scalar)
}

// --- Apple Accelerate NIF ---
//
// macOS ships with the Accelerate framework, which includes:
// - BLAS: cblas_dgemm for matrix multiply. Hand-tuned for Apple silicon.
// - vDSP: Vectorized signal processing. Fast reductions and element-wise ops.
//
// On an M1 Max, cblas_dgemm achieves ~3 TFLOPS FP64.
// Our naive Erlang implementation: ~0.001 TFLOPS. That's 3000x.
//
// The catch: only works on macOS. Falls back gracefully elsewhere.

/// Check if the Apple Accelerate NIF is loaded.
///
/// Returns True on macOS with the NIF built, False elsewhere.
/// Use this to decide whether to use nif_* functions or fall back.
pub fn is_nif_loaded() -> Bool {
  nif_is_loaded_ffi()
}

/// Get backend info string for debugging.
///
/// Returns something like "Apple Accelerate (cblas_dgemm, vDSP)" on macOS.
pub fn nif_backend_info() -> String {
  nif_backend_info_ffi()
}

/// NIF-accelerated matrix multiplication via cblas_dgemm.
///
/// This is where the magic happens on macOS. Apple has spent years
/// optimizing BLAS for their chips. We just call their code.
///
/// Falls back to pure Erlang if NIF not available.
pub fn nif_matmul(
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Result(List(Float), String) {
  nif_matmul_ffi(a, b, m, n, k)
}

/// NIF-accelerated dot product via vDSP.
pub fn nif_dot(a: List(Float), b: List(Float)) -> Result(Float, String) {
  nif_dot_ffi(a, b)
}

/// NIF-accelerated sum via vDSP.
pub fn nif_sum(data: List(Float)) -> Result(Float, String) {
  nif_sum_ffi(data)
}

/// NIF-accelerated scale via vDSP.
pub fn nif_scale(
  data: List(Float),
  scalar: Float,
) -> Result(List(Float), String) {
  nif_scale_ffi(data, scalar)
}

// --- Math Functions ---
//
// Thin wrappers around Erlang :math module.
// These are fine - no need to NIF them, they're already C under the hood.

/// Square root - wraps :math.sqrt/1
pub fn sqrt(x: Float) -> Float {
  sqrt_ffi(x)
}

/// Natural logarithm - wraps :math.log/1
///
/// Undefined for x <= 0. Erlang will return -inf for 0, NaN for negative.
/// Caller's responsibility to check input.
pub fn log(x: Float) -> Float {
  log_ffi(x)
}

/// Exponential e^x - wraps :math.exp/1
///
/// Watch for overflow: exp(710) = inf in Float64.
/// For softmax, subtract max first: exp(x - max(x)).
pub fn exp(x: Float) -> Float {
  exp_ffi(x)
}

/// Cosine - wraps :math.cos/1
pub fn cos(x: Float) -> Float {
  cos_ffi(x)
}

/// Sine - wraps :math.sin/1
pub fn sin(x: Float) -> Float {
  sin_ffi(x)
}

/// Tangent - wraps :math.tan/1
pub fn tan(x: Float) -> Float {
  tan_ffi(x)
}

/// Hyperbolic tangent - wraps :math.tanh/1
///
/// Range: (-1, 1). Saturates for |x| > ~20.
/// Used in some activation functions, though ReLU dominates now.
pub fn tanh(x: Float) -> Float {
  tanh_ffi(x)
}

/// Power x^y - wraps :math.pow/2
pub fn pow(x: Float, y: Float) -> Float {
  pow_ffi(x, y)
}

/// Absolute value.
///
/// Implemented in pure Gleam because :math.abs/1 doesn't exist
/// and erlang:abs/1 is polymorphic (returns same type as input).
pub fn abs(x: Float) -> Float {
  case x <. 0.0 {
    True -> 0.0 -. x
    False -> x
  }
}

// --- Random ---
//
// Erlang's :rand module. Uses a PRNG per process.
// Fine for shuffling, initialization. Not cryptographically secure.

/// Uniform random float in [0, 1).
///
/// Uses Erlang's per-process PRNG (Xoroshiro116+ by default).
/// Not suitable for cryptography, but fine for ML initialization.
///
/// For reproducible results, seed with :rand.seed(Algorithm, Seed).
pub fn random_uniform() -> Float {
  random_uniform_ffi()
}

// --- Constants ---
//
// Because typing 3.14159... every time is error-prone.

/// Pi to 20 decimal places. More than Float64 can represent anyway.
pub const pi: Float = 3.14159265358979323846

/// Euler's number e to 20 decimal places.
pub const e: Float = 2.71828182845904523536

// --- FFI Bindings ---
//
// The actual Erlang external function declarations.
// Keep these private - expose through the typed wrappers above.

@external(erlang, "viva_tensor_ffi", "list_to_array")
fn list_to_array_ffi(lst: List(Float)) -> ErlangArray

@external(erlang, "viva_tensor_ffi", "array_get")
fn array_get_ffi(arr: ErlangArray, index: Int) -> Float

@external(erlang, "viva_tensor_ffi", "array_size")
fn array_size_ffi(arr: ErlangArray) -> Int

@external(erlang, "viva_tensor_ffi", "array_to_list")
fn array_to_list_ffi(arr: ErlangArray) -> List(Float)

@external(erlang, "viva_tensor_ffi", "array_dot")
fn array_dot_ffi(a: ErlangArray, b: ErlangArray) -> Float

@external(erlang, "viva_tensor_ffi", "array_matmul")
fn array_matmul_ffi(
  a: ErlangArray,
  b: ErlangArray,
  m: Int,
  n: Int,
  k: Int,
) -> ErlangArray

@external(erlang, "viva_tensor_ffi", "array_sum")
fn array_sum_ffi(arr: ErlangArray) -> Float

@external(erlang, "viva_tensor_ffi", "array_scale")
fn array_scale_ffi(arr: ErlangArray, scalar: Float) -> ErlangArray

@external(erlang, "math", "sqrt")
fn sqrt_ffi(x: Float) -> Float

@external(erlang, "math", "log")
fn log_ffi(x: Float) -> Float

@external(erlang, "math", "exp")
fn exp_ffi(x: Float) -> Float

@external(erlang, "math", "cos")
fn cos_ffi(x: Float) -> Float

@external(erlang, "math", "sin")
fn sin_ffi(x: Float) -> Float

@external(erlang, "math", "tan")
fn tan_ffi(x: Float) -> Float

@external(erlang, "math", "tanh")
fn tanh_ffi(x: Float) -> Float

@external(erlang, "math", "pow")
fn pow_ffi(x: Float, y: Float) -> Float

@external(erlang, "rand", "uniform")
fn random_uniform_ffi() -> Float

// Apple Accelerate NIF bindings
@external(erlang, "viva_tensor_nif", "is_nif_loaded")
fn nif_is_loaded_ffi() -> Bool

@external(erlang, "viva_tensor_nif", "backend_info")
fn nif_backend_info_ffi() -> String

@external(erlang, "viva_tensor_nif", "matmul")
fn nif_matmul_ffi(
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Result(List(Float), String)

@external(erlang, "viva_tensor_nif", "dot")
fn nif_dot_ffi(a: List(Float), b: List(Float)) -> Result(Float, String)

@external(erlang, "viva_tensor_nif", "sum")
fn nif_sum_ffi(data: List(Float)) -> Result(Float, String)

@external(erlang, "viva_tensor_nif", "scale")
fn nif_scale_ffi(
  data: List(Float),
  scalar: Float,
) -> Result(List(Float), String)

// --- Timing ---
//
// For benchmarking. Microsecond precision is enough for tensor ops.

/// Get current time in microseconds.
///
/// Use for benchmarking: before/after difference gives wall-clock time.
/// For production profiling, use Erlang's :fprof or :eprof instead.
pub fn now_microseconds() -> Int {
  now_microseconds_ffi()
}

@external(erlang, "viva_tensor_ffi", "now_microseconds")
fn now_microseconds_ffi() -> Int

// --- Zig SIMD NIF ---
//
// Our portable high-performance backend. Zig compiles to native code
// with explicit SIMD support. No dependency on platform-specific libraries.
//
// Why Zig?
// 1. Cross-compiles everywhere (Linux, macOS, Windows, ARM, x86)
// 2. Explicit SIMD vectors: @Vector(8, f64) gives us AVX on x86, NEON on ARM
// 3. No runtime, no GC - just fast native code
// 4. Plays nice with Erlang NIFs
//
// Performance: 10-100x vs pure Erlang for vectorizable ops.
// Not quite Apple Accelerate levels, but works everywhere.

/// Check if Zig SIMD NIF is loaded.
pub fn zig_is_loaded() -> Bool {
  zig_is_loaded_ffi()
}

/// Get Zig backend info for debugging.
///
/// Returns SIMD capability info: "Zig SIMD (AVX2)" or "Zig SIMD (NEON)" etc.
pub fn zig_backend_info() -> String {
  zig_backend_info_ffi()
}

/// Zig SIMD dot product.
///
/// Uses 4-way or 8-way SIMD depending on platform.
/// Unrolled loop with accumulator to maximize throughput.
pub fn zig_dot(a: List(Float), b: List(Float)) -> Result(Float, String) {
  zig_dot_ffi(a, b)
}

/// Zig SIMD sum reduction.
pub fn zig_sum(data: List(Float)) -> Result(Float, String) {
  zig_sum_ffi(data)
}

/// Zig SIMD scale (multiply all elements by scalar).
pub fn zig_scale(
  data: List(Float),
  scalar: Float,
) -> Result(List(Float), String) {
  zig_scale_ffi(data, scalar)
}

/// Zig SIMD element-wise add.
pub fn zig_add(a: List(Float), b: List(Float)) -> Result(List(Float), String) {
  zig_add_ffi(a, b)
}

/// Zig SIMD element-wise multiply.
pub fn zig_mul(a: List(Float), b: List(Float)) -> Result(List(Float), String) {
  zig_mul_ffi(a, b)
}

/// Zig SIMD matrix multiplication.
///
/// Tiled implementation with SIMD inner loops.
/// Not quite BLAS-level but respectable: ~10-50 GFLOPS depending on platform.
pub fn zig_matmul(
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Result(List(Float), String) {
  zig_matmul_ffi(a, b, m, n, k)
}

// Zig NIF FFI bindings
@external(erlang, "viva_tensor_zig", "is_loaded")
fn zig_is_loaded_ffi() -> Bool

@external(erlang, "viva_tensor_zig", "backend_info")
fn zig_backend_info_ffi() -> String

@external(erlang, "viva_tensor_zig", "simd_dot")
fn zig_dot_ffi(a: List(Float), b: List(Float)) -> Result(Float, String)

@external(erlang, "viva_tensor_zig", "simd_sum")
fn zig_sum_ffi(data: List(Float)) -> Result(Float, String)

@external(erlang, "viva_tensor_zig", "simd_scale")
fn zig_scale_ffi(
  data: List(Float),
  scalar: Float,
) -> Result(List(Float), String)

@external(erlang, "viva_tensor_zig", "simd_add")
fn zig_add_ffi(a: List(Float), b: List(Float)) -> Result(List(Float), String)

@external(erlang, "viva_tensor_zig", "simd_mul")
fn zig_mul_ffi(a: List(Float), b: List(Float)) -> Result(List(Float), String)

@external(erlang, "viva_tensor_zig", "simd_matmul")
fn zig_matmul_ffi(
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Result(List(Float), String)
