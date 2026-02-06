//// Centralized error types for tensor operations.
////
//// Philosophy: fail fast, fail loud, fail informatively.
////
//// Each error variant carries enough context to debug the issue without
//// printf-debugging. ShapeMismatch tells you both shapes. IndexOutOfBounds
//// tells you the index AND the size. No more "index out of bounds" with no context.
////
//// Why a single error type instead of operation-specific ones?
//// - Simpler API (one Result type everywhere)
//// - Easy to convert to user-facing messages
//// - Pattern matching still works for specific handling

import gleam/int
import gleam/list
import gleam/string

/// All the ways tensor operations can fail.
/// Tried to keep it minimal but expressive.
pub type TensorError {
  /// Shape mismatch between two tensors.
  ///
  /// ## Example
  /// Trying to add [2, 3] tensor with [4, 5] tensor.
  ShapeMismatch(expected: List(Int), got: List(Int))

  /// Invalid shape specification.
  ///
  /// ## Example
  /// Data size doesn't match shape dimensions.
  InvalidShape(reason: String)

  /// Dimension-related error (axis out of bounds, etc.).
  ///
  /// ## Example
  /// Accessing axis 3 on a 2D tensor.
  DimensionError(reason: String)

  /// Broadcasting incompatibility.
  ///
  /// ## Example
  /// Cannot broadcast [2, 3] with [4, 5].
  BroadcastError(shape_a: List(Int), shape_b: List(Int))

  /// Index out of bounds.
  ///
  /// ## Example
  /// Accessing index 10 in tensor of size 5.
  IndexOutOfBounds(index: Int, size: Int)

  /// Invalid dtype for operation.
  ///
  /// ## Example
  /// Using INT8 operation on Float32 tensor.
  DtypeError(reason: String)
}

// --- Formatting -------------------------------------------------------------

/// Human-readable error message. Useful for debugging.
pub fn to_string(error: TensorError) -> String {
  case error {
    ShapeMismatch(expected, got) ->
      "Shape mismatch: expected "
      <> shape_to_string(expected)
      <> ", got "
      <> shape_to_string(got)

    InvalidShape(reason) -> "Invalid shape: " <> reason

    DimensionError(reason) -> "Dimension error: " <> reason

    BroadcastError(a, b) ->
      "Cannot broadcast shapes "
      <> shape_to_string(a)
      <> " and "
      <> shape_to_string(b)

    IndexOutOfBounds(index, size) ->
      "Index "
      <> int.to_string(index)
      <> " out of bounds for size "
      <> int.to_string(size)

    DtypeError(reason) -> "Dtype error: " <> reason
  }
}

/// Pretty-print a shape like [2, 3, 4].
pub fn shape_to_string(shape: List(Int)) -> String {
  "[" <> string.join(list.map(shape, int.to_string), ", ") <> "]"
}
