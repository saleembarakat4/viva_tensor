//// 2:4 Structured Sparsity
////
//// NVIDIA Tensor Cores Structured Sparsity
//// Ampere+ Architecture (RTX 3000/4000, A100, H100)
////
//// CONCEITO:
//// Em cada grupo de 4 elementos, apenas 2 são não-zero
//// = 50% dos elementos são zero, mas em padrão ESTRUTURADO
////
//// POR QUE ESTRUTURADO > ALEATÓRIO:
//// - Sparsity aleatória: difícil de acelerar em hardware
//// - Sparsity estruturada: hardware pode pular zeros eficientemente
////
//// FORMATO DE ARMAZENAMENTO:
//// - 2 valores FP16 (32 bits)
//// - 2-bit máscara indicando posições (4 bits para 4 posições)
//// - Total: 36 bits para 4 elementos = 9 bits/elemento vs 16 bits/elemento
//// - Compressão: ~1.8x
////
//// PERFORMANCE:
//// - 2x throughput em Tensor Cores (pula multiplicações por zero)
//// - Combinado com INT8: 4x speedup total!

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import viva_tensor/tensor.{type Tensor, Tensor}

// ============================================================================
// TIPOS
// ============================================================================

/// Bloco 2:4 (4 elementos com 2 não-zeros)
pub type Sparse24Block {
  Sparse24Block(
    /// Os 2 valores não-zero
    values: #(Float, Float),
    /// Máscara 2-bit indicando posições (0-3 para cada valor)
    positions: #(Int, Int),
  )
}

/// Tensor com esparsidade 2:4
pub type Sparse24Tensor {
  Sparse24Tensor(
    /// Blocos 2:4
    blocks: List(Sparse24Block),
    /// Shape original
    shape: List(Int),
    /// Número de elementos originais
    num_elements: Int,
    /// Memória em bytes
    memory_bytes: Int,
    /// Sparsity real (%)
    sparsity_percent: Float,
  )
}

/// Métricas de poda
pub type PruneMetrics {
  PruneMetrics(
    /// Elementos podados (zerados)
    pruned_count: Int,
    /// Total de elementos
    total_count: Int,
    /// Erro de aproximação
    approximation_error: Float,
    /// Magnitude média mantida
    kept_magnitude_mean: Float,
    /// Magnitude média podada
    pruned_magnitude_mean: Float,
  )
}

// ============================================================================
// PRUNING - Seleciona quais elementos manter
// ============================================================================

/// Aplica poda 2:4: mantém os 2 maiores em cada grupo de 4
/// Estratégia: magnitude (abs) - padrão da NVIDIA
pub fn prune_24_magnitude(t: Tensor) -> Sparse24Tensor {
  let data = tensor.to_list(t)
  let shape = get_tensor_shape(t)
  let num_elements = list.length(data)

  // Divide em grupos de 4
  let groups = list.sized_chunk(data, 4)

  // Para cada grupo, mantém os 2 maiores por magnitude
  let blocks = list.map(groups, fn(group) {
    prune_group_magnitude(pad_group(group))
  })

  // Calcula memória
  // - 2 valores FP16 (4 bytes) + 2 posições 2-bit (1 byte) = 5 bytes por bloco
  // - Cada bloco representa 4 elementos
  // - Original: 4 elementos × 4 bytes = 16 bytes
  let num_blocks = list.length(blocks)
  let memory = num_blocks * 5  // 2×FP16 + 4-bit positions

  Sparse24Tensor(
    blocks: blocks,
    shape: shape,
    num_elements: num_elements,
    memory_bytes: memory,
    sparsity_percent: 50.0,
  )
}

/// Poda um grupo de 4 elementos, retornando Sparse24Block
fn prune_group_magnitude(group: List(Float)) -> Sparse24Block {
  // Indexa elementos com suas magnitudes
  let indexed = list.index_map(group, fn(val, idx) {
    #(idx, val, float.absolute_value(val))
  })

  // Ordena por magnitude decrescente
  let sorted = list.sort(indexed, fn(a, b) {
    float.compare(b.2, a.2)  // Descending by magnitude
  })

  // Pega os 2 maiores
  case sorted {
    [first, second, ..] -> {
      // Ordena por posição para consistência
      let #(pos1, val1, _) = first
      let #(pos2, val2, _) = second
      let #(p1, v1, p2, v2) = case pos1 < pos2 {
        True -> #(pos1, val1, pos2, val2)
        False -> #(pos2, val2, pos1, val1)
      }
      Sparse24Block(values: #(v1, v2), positions: #(p1, p2))
    }
    _ -> Sparse24Block(values: #(0.0, 0.0), positions: #(0, 1))
  }
}

/// Pad grupo para ter exatamente 4 elementos
fn pad_group(group: List(Float)) -> List(Float) {
  let len = list.length(group)
  case len < 4 {
    True -> list.append(group, list.repeat(0.0, 4 - len))
    False -> list.take(group, 4)
  }
}

// ============================================================================
// ALTERNATIVAS DE PRUNING
// ============================================================================

/// Poda baseada em gradiente (para treinamento)
/// Mantém elementos com maior |valor × gradiente|
pub fn prune_24_gradient(
  weights: Tensor,
  gradients: Tensor,
) -> Sparse24Tensor {
  let w_data = tensor.to_list(weights)
  let g_data = tensor.to_list(gradients)

  let shape = get_tensor_shape(weights)
  let num_elements = list.length(w_data)

  // Combina peso × gradiente
  let importance = list.map2(w_data, g_data, fn(w, g) {
    float.absolute_value(w *. g)
  })

  // Agrupa
  let w_groups = list.sized_chunk(w_data, 4)
  let i_groups = list.sized_chunk(importance, 4)

  let blocks = list.map2(w_groups, i_groups, fn(w_group, i_group) {
    prune_group_by_importance(pad_group(w_group), i_group)
  })

  let num_blocks = list.length(blocks)

  Sparse24Tensor(
    blocks: blocks,
    shape: shape,
    num_elements: num_elements,
    memory_bytes: num_blocks * 5,
    sparsity_percent: 50.0,
  )
}

fn prune_group_by_importance(
  weights: List(Float),
  importance: List(Float),
) -> Sparse24Block {
  let indexed = list.zip(list.range(0, 3), list.zip(weights, importance))
    |> list.map(fn(x) {
      let #(idx, #(w, i)) = x
      #(idx, w, i)
    })

  let sorted = list.sort(indexed, fn(a, b) {
    float.compare(b.2, a.2)
  })

  case sorted {
    [first, second, ..] -> {
      let #(pos1, val1, _) = first
      let #(pos2, val2, _) = second
      let #(p1, v1, p2, v2) = case pos1 < pos2 {
        True -> #(pos1, val1, pos2, val2)
        False -> #(pos2, val2, pos1, val1)
      }
      Sparse24Block(values: #(v1, v2), positions: #(p1, p2))
    }
    _ -> Sparse24Block(values: #(0.0, 0.0), positions: #(0, 1))
  }
}

// ============================================================================
// DECOMPRESSÃO
// ============================================================================

/// Reconstrói tensor denso a partir de 2:4 sparse
pub fn decompress(sparse: Sparse24Tensor) -> Tensor {
  let data = list.flat_map(sparse.blocks, fn(block) {
    let #(v1, v2) = block.values
    let #(p1, p2) = block.positions

    // Reconstrói grupo de 4 elementos
    list.range(0, 3)
    |> list.map(fn(i) {
      case i == p1 {
        True -> v1
        False -> case i == p2 {
          True -> v2
          False -> 0.0
        }
      }
    })
  })

  let truncated = list.take(data, sparse.num_elements)
  Tensor(data: truncated, shape: sparse.shape)
}

// ============================================================================
// SPARSE MATMUL (Simulado)
// ============================================================================

/// Matmul com matriz esparsa 2:4
/// Em hardware real (Tensor Cores), isso é 2x mais rápido!
pub fn sparse_matmul(
  sparse_a: Sparse24Tensor,
  dense_b: Tensor,
) -> #(Tensor, Float) {
  // Na implementação real (CUDA), os Tensor Cores pulam as multiplicações por zero
  // Aqui simulamos decomprimindo e multiplicando

  let dense_a = decompress(sparse_a)
  let result = tensor_matmul(dense_a, dense_b)

  // Speedup simulado: 2x (teórico para 2:4 sparsity)
  let theoretical_speedup = 2.0

  #(result, theoretical_speedup)
}

/// Matmul básico para tensors
fn tensor_matmul(a: Tensor, b: Tensor) -> Tensor {
  let a_data = tensor.to_list(a)
  let b_data = tensor.to_list(b)

  let a_shape = get_tensor_shape(a)
  let b_shape = get_tensor_shape(b)

  let #(m, k) = case a_shape {
    [rows, cols] -> #(rows, cols)
    _ -> #(1, list.length(a_data))
  }

  let #(_k2, n) = case b_shape {
    [rows, cols] -> #(rows, cols)
    _ -> #(list.length(b_data), 1)
  }

  let a_rows = list.sized_chunk(a_data, k)
  let b_cols = transpose_matrix(list.sized_chunk(b_data, n))

  let result = list.flat_map(a_rows, fn(a_row) {
    list.map(b_cols, fn(b_col) {
      list.map2(a_row, b_col, fn(x, y) { x *. y })
      |> list.fold(0.0, float.add)
    })
  })

  Tensor(data: result, shape: [m, n])
}

fn transpose_matrix(m: List(List(Float))) -> List(List(Float)) {
  case m {
    [] -> []
    [first, ..] -> {
      let n_cols = list.length(first)
      list.range(0, n_cols - 1)
      |> list.map(fn(col_idx) {
        list.filter_map(m, fn(row) {
          case list.drop(row, col_idx) {
            [x, ..] -> Ok(x)
            [] -> Error(Nil)
          }
        })
      })
    }
  }
}

// ============================================================================
// MÉTRICAS
// ============================================================================

/// Calcula métricas de poda
pub fn compute_metrics(original: Tensor, sparse: Sparse24Tensor) -> PruneMetrics {
  let orig_data = tensor.to_list(original)
  let decomp_data = tensor.to_list(decompress(sparse))

  // Erro de aproximação
  let errors = list.map2(orig_data, decomp_data, fn(o, d) {
    float.absolute_value(o -. d)
  })
  let mean_error = case errors {
    [] -> 0.0
    _ -> list.fold(errors, 0.0, float.add) /. int.to_float(list.length(errors))
  }

  // Magnitudes mantidas vs podadas
  let kept = list.filter_map(list.zip(orig_data, decomp_data), fn(pair) {
    let #(o, d) = pair
    case float.absolute_value(d) >. 0.0 {
      True -> Ok(float.absolute_value(o))
      False -> Error(Nil)
    }
  })

  let pruned = list.filter_map(list.zip(orig_data, decomp_data), fn(pair) {
    let #(o, d) = pair
    case float.absolute_value(d) >. 0.0 {
      False -> Ok(float.absolute_value(o))
      True -> Error(Nil)
    }
  })

  let kept_mean = case kept {
    [] -> 0.0
    _ -> list.fold(kept, 0.0, float.add) /. int.to_float(list.length(kept))
  }

  let pruned_mean = case pruned {
    [] -> 0.0
    _ -> list.fold(pruned, 0.0, float.add) /. int.to_float(list.length(pruned))
  }

  PruneMetrics(
    pruned_count: list.length(pruned),
    total_count: list.length(orig_data),
    approximation_error: mean_error,
    kept_magnitude_mean: kept_mean,
    pruned_magnitude_mean: pruned_mean,
  )
}

// ============================================================================
// BENCHMARK
// ============================================================================

pub fn main() {
  benchmark_sparsity()
}

pub fn benchmark_sparsity() {
  io.println("╔══════════════════════════════════════════════════════════════════╗")
  io.println("║  2:4 STRUCTURED SPARSITY - NVIDIA Tensor Cores                   ║")
  io.println("║  Ampere+ Architecture (RTX 3000/4000, A100, H100)                ║")
  io.println("╚══════════════════════════════════════════════════════════════════╝\n")

  io.println("CONCEITO:")
  io.println("  - Em cada 4 elementos, mantém apenas 2 (50% sparsity)")
  io.println("  - Padrão ESTRUTURADO permite aceleração em hardware")
  io.println("  - Tensor Cores pulam multiplicações por zero")
  io.println("  - Resultado: 2x throughput com ~1% perda de accuracy!\n")

  io.println("FORMATO DE ARMAZENAMENTO:")
  io.println("  - Original: 4 × FP16 = 64 bits")
  io.println("  - Sparse: 2 × FP16 + 4-bit mask = 36 bits")
  io.println("  - Compressão: 1.78x\n")

  // Benchmark
  let t = tensor.random_uniform([1024, 512])

  io.println("━━━ BENCHMARK: Tensor [1024, 512] ━━━")

  let #(time_prune, sparse) = timer_tc(fn() {
    prune_24_magnitude(t)
  })

  let metrics = compute_metrics(t, sparse)

  let original_bytes = 1024 * 512 * 4
  let compression = int.to_float(original_bytes) /. int.to_float(sparse.memory_bytes)

  io.println("  Tempo de poda:      " <> int.to_string(time_prune / 1000) <> "ms")
  io.println("  Memória original:   " <> int.to_string(original_bytes / 1024) <> " KB")
  io.println("  Memória sparse:     " <> int.to_string(sparse.memory_bytes / 1024) <> " KB")
  io.println("  Compressão:         " <> float_to_string(compression) <> "x")
  io.println("  Sparsity:           " <> float_to_string(sparse.sparsity_percent) <> "%")
  io.println("")
  io.println("  Elementos podados:  " <> int.to_string(metrics.pruned_count) <> "/" <>
             int.to_string(metrics.total_count))
  io.println("  Erro aproximação:   " <> float_to_string(metrics.approximation_error))
  io.println("  Magnitude mantida:  " <> float_to_string(metrics.kept_magnitude_mean))
  io.println("  Magnitude podada:   " <> float_to_string(metrics.pruned_magnitude_mean))

  // Simula matmul
  io.println("\n━━━ SPARSE MATMUL SIMULATION ━━━")

  let b = tensor.random_uniform([512, 256])

  let #(time_dense, dense_result) = timer_tc(fn() {
    tensor_matmul(decompress(sparse), b)
  })

  let #(time_sparse, #(sparse_result, speedup)) = timer_tc(fn() {
    sparse_matmul(sparse, b)
  })

  io.println("  Dense matmul:       " <> int.to_string(time_dense / 1000) <> "ms")
  io.println("  Sparse matmul:      " <> int.to_string(time_sparse / 1000) <> "ms (simulado)")
  io.println("  Speedup teórico:    " <> float_to_string(speedup) <> "x (hardware real)")

  // Verificação
  let dense_data = tensor.to_list(dense_result)
  let sparse_data = tensor.to_list(sparse_result)
  let diff = list.map2(dense_data, sparse_data, fn(d, s) {
    float.absolute_value(d -. s)
  }) |> list.fold(0.0, float.max)

  io.println("  Diferença máxima:   " <> float_to_string(diff) <> " (deveria ser ~0)")

  // Comparação com outras técnicas
  io.println("\n━━━ COMPARAÇÃO: COMBINANDO TÉCNICAS ━━━")
  io.println("  FP16:               2x compressão")
  io.println("  INT8:               4x compressão")
  io.println("  2:4 Sparsity:       2x speedup (+ 1.78x compressão)")
  io.println("  NF4:                8x compressão")
  io.println("  ")
  io.println("  INT8 + 2:4:         4x × 1.78x = 7.12x compressão, 8x speedup!")
  io.println("  NF4 + 2:4:          8x × 1.78x = 14.24x compressão!")

  io.println("\n╔══════════════════════════════════════════════════════════════════╗")
  io.println("║  POR QUE 2:4 SPARSITY É ESSENCIAL:                               ║")
  io.println("║                                                                  ║")
  io.println("║  1. Hardware nativo em RTX 3000/4000/A100/H100                   ║")
  io.println("║  2. 2x throughput com ~1% perda de accuracy                      ║")
  io.println("║  3. Combina com quantização para 4x+ total                       ║")
  io.println("║  4. Padrão em modelos NVIDIA (Megatron-LM, etc)                  ║")
  io.println("║                                                                  ║")
  io.println("║  viva_tensor + 2:4 = Máximo uso dos Tensor Cores!                ║")
  io.println("╚══════════════════════════════════════════════════════════════════╝")
}

// ============================================================================
// HELPERS
// ============================================================================

fn float_to_string(f: Float) -> String {
  let rounded = int.to_float(float.round(f *. 10000.0)) /. 10000.0
  float.to_string(rounded)
}

fn get_tensor_shape(t: Tensor) -> List(Int) {
  case t {
    Tensor(_, shape) -> shape
    tensor.StridedTensor(_, shape, _, _) -> shape
  }
}

@external(erlang, "timer", "tc")
fn timer_tc(f: fn() -> a) -> #(Int, a)
