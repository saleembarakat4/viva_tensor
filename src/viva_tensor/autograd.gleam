import gleam/dict.{type Dict}
import gleam/int
import gleam/list
import gleam/result
import gleam/string
import viva_tensor/tensor.{type Tensor}

/// Unique identifier for each node in the computational graph
pub type NodeId =
  Int

/// Function that calculates parent gradients based on this node's gradient
pub type BackwardFn =
  fn(Tensor) -> List(#(NodeId, Tensor))

/// The Tape maintains the history of operations in the computational graph.
/// Unlike PyTorch (global/mutable), here the Tape is an explicit immutable value.
pub type Tape {
  Tape(
    next_id: NodeId,
    // Maps resulting node ID -> Closure that computes gradients for parents
    operations: Dict(NodeId, BackwardFn),
  )
}

/// A variable tracked in the autograd system
pub type Variable {
  Variable(id: NodeId, data: Tensor)
}

/// The result of a traced operation.
/// Encapsulates the produced value (e.g., Variable) and the new tape state.
/// Analogous to a State Monad.
pub type Traced(a) {
  Traced(value: a, tape: Tape)
}

/// Creates a new empty tape
pub fn new_tape() -> Tape {
  Tape(next_id: 0, operations: dict.new())
}

/// Registers a new variable (leaf node) in the graph
pub fn new_variable(tape: Tape, data: Tensor) -> Traced(Variable) {
  let id = tape.next_id
  let var = Variable(id: id, data: data)
  let new_tape = Tape(..tape, next_id: id + 1)
  Traced(value: var, tape: new_tape)
}

// =============================================================================
// TRACED OPERATIONS
// =============================================================================

/// Operation sequencing (Monadic Pipe)
/// Allows chaining layers: x |> sequence(layer1) |> sequence(layer2)
pub fn sequence(
  input: Result(Traced(Variable), e),
  layer_fn: fn(Tape, Variable) -> Result(Traced(Variable), e),
) -> Result(Traced(Variable), e) {
  use Traced(var, tape) <- result.try(input)
  layer_fn(tape, var)
}

/// Traced addition: c = a + b (supports broadcasting)
pub fn add(
  tape: Tape,
  a: Variable,
  b: Variable,
) -> Result(Traced(Variable), tensor.TensorError) {
  use res_data <- result.try(tensor.add_broadcast(a.data, b.data))

  let res_id = tape.next_id

  // Backward: y = a + b
  // If broadcasting occurred, we need to sum gradients over expanded dimensions
  let backward = fn(grad: Tensor) {
    let grad_a = case grad.shape == a.data.shape {
      True -> grad
      False -> {
        // Simplified reduction: if rank differs or dimension is 1, sum
        // In MVP, if a is [4] and grad is [5, 4], sum_axis(0) solves it
        let assert Ok(res) = tensor.sum_axis(grad, 0)
        res
      }
    }

    let grad_b = case grad.shape == b.data.shape {
      True -> grad
      False -> {
        let assert Ok(res) = tensor.sum_axis(grad, 0)
        res
      }
    }

    [#(a.id, grad_a), #(b.id, grad_b)]
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// Traced subtraction: c = a - b
pub fn sub(
  tape: Tape,
  a: Variable,
  b: Variable,
) -> Result(Traced(Variable), tensor.TensorError) {
  use res_data <- result.try(tensor.sub(a.data, b.data))

  let res_id = tape.next_id

  // Backward: y = a - b 
  // dy/da = 1 * grad
  // dy/db = -1 * grad
  let backward = fn(grad: Tensor) {
    let neg_grad = tensor.negate(grad)
    [#(a.id, grad), #(b.id, neg_grad)]
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// Traced Element-wise Multiplication: c = a * b
pub fn mul(
  tape: Tape,
  a: Variable,
  b: Variable,
) -> Result(Traced(Variable), tensor.TensorError) {
  use res_data <- result.try(tensor.mul(a.data, b.data))

  let res_id = tape.next_id

  // Backward: y = a * b
  // dy/da = b * grad
  // dy/db = a * grad
  let backward = fn(grad: Tensor) {
    // TODO: Handle errors
    let assert Ok(grad_a) = tensor.mul(grad, b.data)
    let assert Ok(grad_b) = tensor.mul(grad, a.data)
    [#(a.id, grad_a), #(b.id, grad_b)]
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// Traced Mean (Reduce Mean): y = mean(x)
/// Returns a scalar Tensor (rank 0 or 1 depending on base implementation, here we force 1)
pub fn mean(tape: Tape, a: Variable) -> Traced(Variable) {
  let val = tensor.mean(a.data)
  let res_data = tensor.from_list([val])
  // Wrap scalar in tensor

  let res_id = tape.next_id

  // Backward: y = sum(x) / n
  // dy/dx = (1/n) * grad
  // The input gradient (grad) is a scalar (or 1-element tensor)
  // We need to expand it to x's shape and divide by n
  let backward = fn(grad: Tensor) {
    let n = tensor.size(a.data) |> int.to_float
    let grad_val = tensor.to_list(grad) |> list.first |> result.unwrap(1.0)
    let scaled_grad_val = grad_val /. n

    // Creates a filled tensor with the scaled gradient value
    let grad_input = tensor.fill(a.data.shape, scaled_grad_val)
    [#(a.id, grad_input)]
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Traced(value: Variable(id: res_id, data: res_data), tape: new_tape)
}

/// Traced Matrix Multiplication: c = a @ b
pub fn matmul(
  tape: Tape,
  a: Variable,
  b: Variable,
) -> Result(Traced(Variable), tensor.TensorError) {
  use res_data <- result.try(tensor.matmul(a.data, b.data))

  let res_id = tape.next_id

  // Backward: y = a @ b
  // dy/da = grad @ b.T
  // dy/db = a.T @ grad
  let backward = fn(grad: Tensor) {
    // TODO: Robust error handling inside backward
    let assert Ok(bt) = tensor.transpose(b.data)
    let assert Ok(at) = tensor.transpose(a.data)

    let assert Ok(grad_a) = tensor.matmul(grad, bt)
    let assert Ok(grad_b) = tensor.matmul(at, grad)

    [#(a.id, grad_a), #(b.id, grad_b)]
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// Traced Transpose: c = a.T
pub fn transpose(
  tape: Tape,
  a: Variable,
) -> Result(Traced(Variable), tensor.TensorError) {
  use res_data <- result.try(tensor.transpose(a.data))

  let res_id = tape.next_id

  // Backward: y = a.T => dy/da = grad.T
  let backward = fn(grad: Tensor) {
    let assert Ok(grad_t) = tensor.transpose(grad)
    [#(a.id, grad_t)]
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// Traced ReLU
pub fn relu(tape: Tape, a: Variable) -> Traced(Variable) {
  let res_data =
    tensor.map(a.data, fn(x) {
      case x >. 0.0 {
        True -> x
        False -> 0.0
      }
    })

  let res_id = tape.next_id

  // Backward: y = relu(a) => dy/da = 1 if a > 0 else 0
  let backward = fn(grad: Tensor) {
    let mask =
      tensor.map(a.data, fn(x) {
        case x >. 0.0 {
          True -> 1.0
          False -> 0.0
        }
      })
    let assert Ok(grad_a) = tensor.mul(grad, mask)
    [#(a.id, grad_a)]
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Traced(value: Variable(id: res_id, data: res_data), tape: new_tape)
}

// =============================================================================
// BACKPROPAGATION ENGINE
// =============================================================================

/// Executes backpropagation starting from a scalar variable (loss).
/// Returns a Map of NodeId -> Gradient (Tensor)
pub fn backward(
  tape: Tape,
  loss: Variable,
) -> Result(Dict(NodeId, Tensor), String) {
  // Initial gradient dLoss/dLoss = 1.0
  let initial_grad = tensor.ones(loss.data.shape)
  let initial_grads = dict.from_list([#(loss.id, initial_grad)])

  // Process nodes in reverse creation order (Implicit Topological Sort)
  // IDs are sequential, so (next_id - 1) down to 0 ensures correct order.
  let all_ids = list.range(tape.next_id - 1, 0)

  let final_grads =
    list.fold(all_ids, initial_grads, fn(grads, current_id) {
      case dict.get(grads, current_id) {
        Error(_) -> grads
        // Node does not contribute to loss or hasn't been computed
        Ok(current_grad) -> {
          // If this node has an operation registered on the tape, expand gradient
          case dict.get(tape.operations, current_id) {
            Error(_) -> grads
            // Leaf node (input), no parents to propagate to
            Ok(back_fn) -> {
              let parent_grads = back_fn(current_grad)

              // Accumulate gradients in parents (sum if gradient already exists)
              list.fold(parent_grads, grads, fn(acc_grads, pair) {
                let #(pid, pgrad) = pair
                case dict.get(acc_grads, pid) {
                  Error(_) -> dict.insert(acc_grads, pid, pgrad)
                  Ok(existing) -> {
                    case existing.shape == pgrad.shape {
                      True -> {
                        let assert Ok(sum) = tensor.add(existing, pgrad)
                        dict.insert(acc_grads, pid, sum)
                      }
                      False -> {
                        let msg =
                          "ShapeMismatch at node "
                          <> int.to_string(pid)
                          <> ": existing="
                          <> string_shape(existing.shape)
                          <> ", new="
                          <> string_shape(pgrad.shape)
                        panic as msg
                      }
                    }
                  }
                }
              })
            }
          }
        }
      }
    })

  Ok(final_grads)
}

fn string_shape(shape: List(Int)) -> String {
  "[" <> string.join(list.map(shape, int.to_string), with: ", ") <> "]"
}
