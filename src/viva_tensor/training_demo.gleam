import gleam/dict
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/result
import viva_tensor/autograd.{type Tape, type Variable, Traced}
import viva_tensor/nn
import viva_tensor/tensor

// Configurações do treinamento
const learning_rate = 0.01

const epochs = 500

const input_features = 1

const hidden_features = 4

const output_features = 1

pub type TrainingState {
  TrainingState(tape: Tape, layer1: nn.Linear, layer2: nn.Linear)
}

pub fn main() {
  io.println("🚀 Iniciando Demonstração de Treinamento Mycelial...")

  // 1. Criar Dados de Treinamento
  let tape = autograd.new_tape()

  // Reshape para [5, 1] (5 amostras, 1 feature)
  let x_data = tensor.from_list([1.0, 2.0, 3.0, 4.0, 5.0])
  let assert Ok(x_data) = tensor.reshape(x_data, [5, 1])

  let y_data = tensor.from_list([2.1, 3.9, 6.2, 8.1, 10.3])
  let assert Ok(y_data) = tensor.reshape(y_data, [5, 1])

  let Traced(x, tape1) = autograd.new_variable(tape, x_data)
  let Traced(y, tape2) = autograd.new_variable(tape1, y_data)

  // 2. Inicializar Camadas
  let Traced(layer1, tape3) = nn.linear(tape2, input_features, hidden_features)
  let Traced(layer2, tape4) = nn.linear(tape3, hidden_features, output_features)

  let state = TrainingState(tape: tape4, layer1: layer1, layer2: layer2)

  // 3. Rodar Loop de Treinamento
  let _final_state =
    list.fold(list.range(0, epochs - 1), state, fn(acc_state, epoch) {
      train_step(acc_state, epoch, x_data, y_data)
    })

  io.println("✅ Treinamento finalizado!")
}

fn train_step(
  state: TrainingState,
  epoch: Int,
  x_data: tensor.Tensor,
  y_data: tensor.Tensor,
) -> TrainingState {
  // Re-registra x e y na fita atual
  let Traced(x, tape1) = autograd.new_variable(state.tape, x_data)
  let Traced(target, tape2) = autograd.new_variable(tape1, y_data)

  // Forward Pass
  let assert Ok(Traced(l1_out, tape3)) =
    nn.linear_forward(tape2, state.layer1, x)
  let Traced(hidden_act, tape4) = nn.relu(tape3, l1_out)
  let assert Ok(Traced(output, tape5)) =
    nn.linear_forward(tape4, state.layer2, hidden_act)

  // Calcular Loss
  let assert Ok(Traced(loss_var, tape6)) = nn.mse_loss(tape5, output, target)

  // Backward Pass
  let assert Ok(grads) = autograd.backward(tape6, loss_var)

  // Log progress
  case epoch % 100 == 0 {
    True -> {
      let loss_val =
        tensor.to_list(loss_var.data) |> list.first |> result.unwrap(0.0)
      io.println(
        "Época "
        <> int.to_string(epoch)
        <> " | Loss: "
        <> float.to_string(loss_val),
      )
    }
    False -> Nil
  }

  // Atualizar Parâmetros (Gradient Descent Manual)
  let assert Ok(gw1) = dict.get(grads, state.layer1.w.id)
  let assert Ok(gb1) = dict.get(grads, state.layer1.b.id)
  let assert Ok(new_w1_data) =
    tensor.sub(state.layer1.w.data, tensor.scale(gw1, learning_rate))
  let assert Ok(new_b1_data) =
    tensor.sub(state.layer1.b.data, tensor.scale(gb1, learning_rate))

  let assert Ok(gw2) = dict.get(grads, state.layer2.w.id)
  let assert Ok(gb2) = dict.get(grads, state.layer2.b.id)
  let assert Ok(new_w2_data) =
    tensor.sub(state.layer2.w.data, tensor.scale(gw2, learning_rate))
  let assert Ok(new_b2_data) =
    tensor.sub(state.layer2.b.data, tensor.scale(gb2, learning_rate))

  // Nova fita para a próxima iteração
  let next_tape = autograd.new_tape()

  // Registrar novos pesos
  let Traced(nw1, nt1) = autograd.new_variable(next_tape, new_w1_data)
  let Traced(nb1, nt2) = autograd.new_variable(nt1, new_b1_data)
  let Traced(nw2, nt3) = autograd.new_variable(nt2, new_w2_data)
  let Traced(nb2, nt4) = autograd.new_variable(nt3, new_b2_data)

  TrainingState(
    tape: nt4,
    layer1: nn.Linear(w: nw1, b: nb1),
    layer2: nn.Linear(w: nw2, b: nb2),
  )
}
