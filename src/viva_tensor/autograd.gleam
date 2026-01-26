import gleam/dict.{type Dict}
import gleam/list
import gleam/result
import viva_tensor/tensor.{type Tensor}

/// Identificador único para cada nó no grafo computacional
pub type NodeId =
  Int

/// Função que calcula o gradiente dos pais com base no gradiente deste nó
pub type BackwardFn =
  fn(Tensor) -> List(#(NodeId, Tensor))

/// O contexto mantém o estado do grafo computacional (a "fita")
pub type Context {
  Context(
    next_id: NodeId,
    // A fita armazena a função backward associada ao ID do nó resultante
    tape: Dict(NodeId, BackwardFn),
  )
}

/// Uma variável rastreada no autograd
pub type Variable {
  Variable(id: NodeId, data: Tensor)
}

/// Cria um novo contexto vazio
pub fn new_context() -> Context {
  Context(next_id: 0, tape: dict.new())
}

/// Cria uma nova variável (folha) no grafo
pub fn new_variable(ctx: Context, data: Tensor) -> #(Context, Variable) {
  let id = ctx.next_id
  let var = Variable(id: id, data: data)
  let new_ctx = Context(..ctx, next_id: id + 1)
  #(new_ctx, var)
}

// =============================================================================
// OPERAÇÕES DIFERENCIÁVEIS
// =============================================================================

pub fn add(ctx: Context, a: Variable, b: Variable) -> Result(#(Context, Variable), tensor.TensorError) {
  use res_data <- result.try(tensor.add(a.data, b.data))
  
  let res_id = ctx.next_id
  
  // Gradiente da soma é distribuído igualmente: 
  // y = a + b => dy/da = 1, dy/db = 1
  // grad_a = grad_y * 1
  let backward = fn(grad: Tensor) {
    [#(a.id, grad), #(b.id, grad)]
  }
  
  let new_tape = dict.insert(ctx.tape, res_id, backward)
  let new_ctx = Context(next_id: res_id + 1, tape: new_tape)
  
  Ok(#(new_ctx, Variable(id: res_id, data: res_data)))
}

pub fn matmul(ctx: Context, a: Variable, b: Variable) -> Result(#(Context, Variable), tensor.TensorError) {
  use res_data <- result.try(tensor.matmul(a.data, b.data))
  
  let res_id = ctx.next_id
  
  // y = a @ b
  // dy/da = grad @ b.T
  // dy/db = a.T @ grad
  let backward = fn(grad: Tensor) {
    // TODO: Tratamento de erro nos backwards (assumindo sucesso por enquanto ou panic safe)
    let assert Ok(bt) = tensor.transpose(b.data)
    let assert Ok(at) = tensor.transpose(a.data)
    
    let assert Ok(grad_a) = tensor.matmul(grad, bt)
    let assert Ok(grad_b) = tensor.matmul(at, grad)
    
    [#(a.id, grad_a), #(b.id, grad_b)]
  }
  
  let new_tape = dict.insert(ctx.tape, res_id, backward)
  let new_ctx = Context(next_id: res_id + 1, tape: new_tape)
  
  Ok(#(new_ctx, Variable(id: res_id, data: res_data)))
}

pub fn transpose(ctx: Context, a: Variable) -> Result(#(Context, Variable), tensor.TensorError) {
  use res_data <- result.try(tensor.transpose(a.data))
  
  let res_id = ctx.next_id
  
  // y = a.T
  // dy/da = grad.T
  let backward = fn(grad: Tensor) {
    let assert Ok(grad_t) = tensor.transpose(grad)
    [#(a.id, grad_t)]
  }
  
  let new_tape = dict.insert(ctx.tape, res_id, backward)
  let new_ctx = Context(next_id: res_id + 1, tape: new_tape)
  
  Ok(#(new_ctx, Variable(id: res_id, data: res_data)))
}

pub fn relu(ctx: Context, a: Variable) -> #(Context, Variable) {
  let res_data = tensor.map(a.data, fn(x) { 
    case x >. 0.0 {
      True -> x
      False -> 0.0
    }
  })
  
  let res_id = ctx.next_id
  
  // y = relu(a)
  // dy/da = 1 se a > 0 senão 0
  let backward = fn(grad: Tensor) {
    let mask = tensor.map(a.data, fn(x) {
      case x >. 0.0 {
        True -> 1.0
        False -> 0.0
      }
    })
    let assert Ok(grad_a) = tensor.mul(grad, mask)
    [#(a.id, grad_a)]
  }
  
  let new_tape = dict.insert(ctx.tape, res_id, backward)
  let new_ctx = Context(next_id: res_id + 1, tape: new_tape)
  
  #(new_ctx, Variable(id: res_id, data: res_data))
}

// =============================================================================
// BACKPROPAGATION ENGINE
// =============================================================================

/// Executa o backpropagation a partir de uma variável escalar (loss)
/// Retorna um Mapa de NodeId -> Gradiente
pub fn backward(ctx: Context, loss: Variable) -> Result(Dict(NodeId, Tensor), String) {
  // O gradiente inicial da loss em relação a ela mesma é 1.0
  let initial_grad = tensor.ones(loss.data.shape)
  let initial_grads = dict.from_list([#(loss.id, initial_grad)])
  
  // Processar nós na ordem reversa de criação (topological sort implícito pela ordem de IDs)
  // IDs vão de 0 a next_id - 1. Backprop vai de next_id - 1 até 0.
  let all_ids = list.range(ctx.next_id - 1, 0)
  
  let final_grads = list.fold(all_ids, initial_grads, fn(grads, current_id) {
    case dict.get(grads, current_id) {
      Error(_) -> grads // Nó não contribui para a loss ou não foi computado
      Ok(current_grad) -> {
        // Se este nó tem uma função backward na fita, chame-a
        case dict.get(ctx.tape, current_id) {
          Error(_) -> grads // Nó folha (input), não tem backward function
          Ok(back_fn) -> {
            let parent_grads = back_fn(current_grad)
            // Acumular gradientes nos pais
            list.fold(parent_grads, grads, fn(acc_grads, pair) {
              let #(pid, pgrad) = pair
              case dict.get(acc_grads, pid) {
                Error(_) -> dict.insert(acc_grads, pid, pgrad)
                Ok(existing) -> {
                  let assert Ok(sum) = tensor.add(existing, pgrad)
                  dict.insert(acc_grads, pid, sum)
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
