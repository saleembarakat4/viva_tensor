import gleam/result
import viva_tensor/autograd.{type Context, type Variable}
import viva_tensor/tensor

pub type Linear {
  Linear(w: Variable, b: Variable)
}

/// Cria uma nova camada Linear
pub fn linear(ctx: Context, in_features: Int, out_features: Int) -> #(Context, Linear) {
  // Inicialização de pesos (Xavier/Glorot)
  let w_data = tensor.xavier_init(in_features, out_features)
  let b_data = tensor.zeros([out_features])
  
  let #(ctx1, w) = autograd.new_variable(ctx, w_data)
  let #(ctx2, b) = autograd.new_variable(ctx1, b_data)
  
  #(ctx2, Linear(w, b))
}

/// Forward pass da camada Linear
/// y = x @ w.T + b
pub fn linear_forward(ctx: Context, layer: Linear, x: Variable) -> Result(#(Context, Variable), tensor.TensorError) {
  // Transpor pesos: [out, in] -> [in, out]
  use #(ctx1, wt) <- result.try(autograd.transpose(ctx, layer.w))
  
  // Matmul: [batch, in] @ [in, out] -> [batch, out]
  use #(ctx2, xw) <- result.try(autograd.matmul(ctx1, x, wt))
  
  // Add Bias: [batch, out] + [out] (requer broadcast)
  // Por enquanto usamos add normal, assumindo shapes compatíveis ou broadcast implícito futuro
  autograd.add(ctx2, xw, layer.b)
}

/// Módulo de funções de ativação
pub fn relu(ctx: Context, x: Variable) -> #(Context, Variable) {
  autograd.relu(ctx, x)
}

/// Loss function: Mean Squared Error
pub fn mse_loss(ctx: Context, pred: Variable, target: Variable) -> Result(#(Context, Variable), tensor.TensorError) {
  // diff = pred - target
  // square = diff * diff
  // mean = sum(square) / size
  
  // Precisamos implementar sub, mul, mean no autograd.
  // Como MVP, vamos retornar um erro NotImplemented se tentar usar sem essas ops
  Error(tensor.DimensionError("MSE Loss requer implementação de sub/mul/mean no autograd"))
}
