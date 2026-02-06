//// Auto-Tuning System - Self-Optimizing Tensor Library
////
//// Inspired by HuggingChat + ggml + Candle research
////
//// Features:
//// 1. GPU Auto-Detection (detects VRAM, load, capabilities)
//// 2. Adaptive Quantization (int8 for inference, fp32 for training)
//// 3. Zero-Copy between Gleam and Rust NIFs
//// 4. Auto Batch Size Optimizer (learns the best batch for the hardware)
////
//// Target: RTX 4090 24GB VRAM + 32GB RAM

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import viva_tensor/tensor.{type Tensor}

// ============================================================================
// TYPES - HARDWARE DETECTION
// ============================================================================

/// Detected device type
pub type Device {
  Cuda(gpu_id: Int, vram_gb: Float)
  Metal(device_id: Int)
  Cpu(cores: Int)
}

/// Complete hardware profile
pub type HardwareProfile {
  HardwareProfile(
    device: Device,
    total_vram_gb: Float,
    available_vram_gb: Float,
    total_ram_gb: Float,
    gpu_load_pct: Float,
    optimal_batch_size: Int,
  )
}

/// Quantization mode
pub type QuantMode {
  Inference
  // INT8 - 4x less memory, faster
  Training
  // FP32 - full precision
  Adaptive
  // Chooses automatically
}

/// Auto-tuner state
pub type AutoTuner {
  AutoTuner(
    hardware: HardwareProfile,
    quant_mode: QuantMode,
    history: List(BatchResult),
    current_batch_size: Int,
  )
}

/// Result of a batch execution
pub type BatchResult {
  BatchResult(batch_size: Int, duration_ms: Float, throughput: Float)
}

// ============================================================================
// HARDWARE DETECTION
// ============================================================================

/// Detects available hardware
pub fn detect_hardware() -> HardwareProfile {
  // For now returns the GATO-PC profile
  // In production: would call NIF for gfxinfo/nvml
  HardwareProfile(
    device: Cuda(gpu_id: 0, vram_gb: 24.0),
    total_vram_gb: 24.0,
    available_vram_gb: 20.0,
    // 4GB reserved for system
    total_ram_gb: 32.0,
    gpu_load_pct: 0.0,
    optimal_batch_size: 32,
  )
}

/// Creates auto-tuner for CPU-only
pub fn detect_cpu_only() -> HardwareProfile {
  HardwareProfile(
    device: Cpu(cores: 16),
    total_vram_gb: 0.0,
    available_vram_gb: 0.0,
    total_ram_gb: 32.0,
    gpu_load_pct: 0.0,
    optimal_batch_size: 8,
  )
}

// ============================================================================
// AUTO-TUNER
// ============================================================================

/// Creates new auto-tuner
pub fn new() -> AutoTuner {
  let hardware = detect_hardware()
  let batch_size = get_default_batch_size(hardware.device)

  AutoTuner(
    hardware: hardware,
    quant_mode: Adaptive,
    history: [],
    current_batch_size: batch_size,
  )
}

/// Default batch size based on device
fn get_default_batch_size(device: Device) -> Int {
  case device {
    Cuda(_, vram_gb) -> {
      // Scales with available VRAM
      case vram_gb >=. 20.0 {
        True -> 64
        // RTX 4090 with plenty of VRAM
        False ->
          case vram_gb >=. 10.0 {
            True -> 32
            // RTX 3080/4080
            False -> 16
            // Smaller GPUs
          }
      }
    }
    Metal(_) -> 16
    // Apple Silicon is efficient with smaller batches
    Cpu(_) -> 8
    // BEAM CPU-only
  }
}

/// Records execution result and optimizes
pub fn profile(
  tuner: AutoTuner,
  batch_size: Int,
  duration_ms: Float,
) -> AutoTuner {
  let throughput = int.to_float(batch_size) /. duration_ms

  let result =
    BatchResult(
      batch_size: batch_size,
      duration_ms: duration_ms,
      throughput: throughput,
    )

  // Add to history (keep last 20)
  let new_history = case list.length(tuner.history) >= 20 {
    True -> [result, ..list.take(tuner.history, 19)]
    False -> [result, ..tuner.history]
  }

  // Find optimal batch size
  let optimal = find_optimal_batch_size(new_history, tuner.hardware)

  AutoTuner(..tuner, history: new_history, current_batch_size: optimal)
}

/// Finds batch size with highest throughput
fn find_optimal_batch_size(
  history: List(BatchResult),
  hw: HardwareProfile,
) -> Int {
  case history {
    [] -> get_default_batch_size(hw.device)
    _ -> {
      // Group by batch size and compute average throughput
      let best =
        history
        |> list.fold(BatchResult(0, 0.0, 0.0), fn(acc, r) {
          case r.throughput >. acc.throughput {
            True -> r
            False -> acc
          }
        })

      // Adjust for available memory
      adjust_for_memory(best.batch_size, hw)
    }
  }
}

/// Adjusts batch size for available memory
fn adjust_for_memory(batch_size: Int, hw: HardwareProfile) -> Int {
  let tensor_size_mb = 2.0
  // ~2MB per tensor 512d fp32
  let batch_memory_gb = int.to_float(batch_size) *. tensor_size_mb /. 1024.0

  // Leave 20% headroom
  let max_memory = hw.available_vram_gb *. 0.8

  case batch_memory_gb >. max_memory {
    True -> {
      let adjusted =
        float.round(max_memory /. tensor_size_mb *. 1024.0)
        |> int.max(1)
      adjusted
    }
    False -> batch_size
  }
}

// ============================================================================
// ADAPTIVE QUANTIZATION
// ============================================================================

/// Quantization context
pub type QuantContext {
  QuantContext(mode: QuantMode, scales: List(Float))
}

/// Creates quantization context
pub fn new_quant_context(mode: QuantMode) -> QuantContext {
  QuantContext(mode: mode, scales: [])
}

/// Decides quantization mode based on the operation
pub fn should_quantize(ctx: QuantContext, is_inference: Bool) -> Bool {
  case ctx.mode {
    Inference -> True
    Training -> False
    Adaptive -> is_inference
  }
}

/// Computes scale for absmax quantization (int8)
pub fn compute_scale(tensor: Tensor) -> Float {
  let max_val = tensor.max(tensor)
  case max_val >. 0.0 {
    True -> 127.0 /. max_val
    False -> 1.0
  }
}

// ============================================================================
// MEMORY PRESSURE HANDLING
// ============================================================================

/// Checks memory pressure
pub fn check_memory_pressure(hw: HardwareProfile) -> MemoryPressure {
  let usage_pct = 1.0 -. { hw.available_vram_gb /. hw.total_vram_gb }

  case usage_pct {
    p if p >=. 0.9 -> Critical
    p if p >=. 0.7 -> High
    p if p >=. 0.5 -> Medium
    _ -> Low
  }
}

pub type MemoryPressure {
  Low
  Medium
  High
  Critical
}

/// Strategy based on memory pressure
pub fn get_memory_strategy(pressure: MemoryPressure) -> MemoryStrategy {
  case pressure {
    Critical ->
      MemoryStrategy(
        batch_size_mult: 0.25,
        quant_mode: Inference,
        gc_aggressive: True,
      )
    High ->
      MemoryStrategy(
        batch_size_mult: 0.5,
        quant_mode: Inference,
        gc_aggressive: True,
      )
    Medium ->
      MemoryStrategy(
        batch_size_mult: 0.75,
        quant_mode: Adaptive,
        gc_aggressive: False,
      )
    Low ->
      MemoryStrategy(
        batch_size_mult: 1.0,
        quant_mode: Training,
        gc_aggressive: False,
      )
  }
}

pub type MemoryStrategy {
  MemoryStrategy(
    batch_size_mult: Float,
    quant_mode: QuantMode,
    gc_aggressive: Bool,
  )
}

// ============================================================================
// BENCHMARK AND PROFILE
// ============================================================================

/// Runs complete hardware profile
pub fn run_hardware_profile() {
  io.println(
    "╔══════════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║  AUTO-TUNE HARDWARE PROFILE                                     ║",
  )
  io.println(
    "╚══════════════════════════════════════════════════════════════════╝\n",
  )

  let hw = detect_hardware()

  io.println("DETECTED HARDWARE:")
  print_device(hw.device)
  io.println("  Total VRAM: " <> float_to_string(hw.total_vram_gb) <> " GB")
  io.println(
    "  Available VRAM: " <> float_to_string(hw.available_vram_gb) <> " GB",
  )
  io.println("  Total RAM: " <> float_to_string(hw.total_ram_gb) <> " GB")
  io.println("  GPU Load: " <> float_to_string(hw.gpu_load_pct) <> "%")
  io.println("  Optimal Batch Size: " <> int.to_string(hw.optimal_batch_size))

  let pressure = check_memory_pressure(hw)
  io.println("\nMEMORY PRESSURE: " <> pressure_to_string(pressure))

  let strategy = get_memory_strategy(pressure)
  io.println("STRATEGY:")
  io.println("  Batch Mult: " <> float_to_string(strategy.batch_size_mult))
  io.println("  Quant Mode: " <> quant_mode_to_string(strategy.quant_mode))
  io.println("  Aggressive GC: " <> bool_to_string(strategy.gc_aggressive))

  io.println(
    "\n╔══════════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║  RECOMMENDATIONS FOR RTX 4090 24GB + 32GB RAM:                  ║",
  )
  io.println(
    "║                                                                  ║",
  )
  io.println(
    "║  1. Batch Size: 64 (can go up to 128 with INT8)                 ║",
  )
  io.println(
    "║  2. Quantization: INT8 for inference (4x less VRAM)             ║",
  )
  io.println(
    "║  3. Memory Pool: Pre-allocate 20GB for tensors                   ║",
  )
  io.println(
    "║  4. Zero-Copy: Use Binary refs between Gleam and Rust            ║",
  )
  io.println(
    "╚══════════════════════════════════════════════════════════════════╝",
  )
}

pub fn main() {
  run_hardware_profile()
}

// ============================================================================
// HELPERS
// ============================================================================

fn print_device(device: Device) {
  case device {
    Cuda(id, vram) ->
      io.println(
        "  Device: CUDA GPU #"
        <> int.to_string(id)
        <> " ("
        <> float_to_string(vram)
        <> "GB)",
      )
    Metal(id) -> io.println("  Device: Metal #" <> int.to_string(id))
    Cpu(cores) ->
      io.println("  Device: CPU (" <> int.to_string(cores) <> " cores)")
  }
}

fn pressure_to_string(p: MemoryPressure) -> String {
  case p {
    Low -> "LOW (all good)"
    Medium -> "MEDIUM (monitoring)"
    High -> "HIGH (reduce batch)"
    Critical -> "CRITICAL (emergency mode)"
  }
}

fn quant_mode_to_string(m: QuantMode) -> String {
  case m {
    Inference -> "INT8 (inference)"
    Training -> "FP32 (training)"
    Adaptive -> "ADAPTIVE"
  }
}

fn bool_to_string(b: Bool) -> String {
  case b {
    True -> "Yes"
    False -> "No"
  }
}

fn float_to_string(f: Float) -> String {
  let rounded = int.to_float(float.round(f *. 100.0)) /. 100.0
  float.to_string(rounded)
}
