package kannoo.core

import kannoo.math.TensorBase

abstract class ElementWiseActivationFunction : ActivationFunction {
    abstract fun compute(x: Float): Float
    abstract fun derivative(x: Float): Float

    final override fun compute(tensor: TensorBase): TensorBase =
        tensor.map(::compute)

    final override fun derivative(tensor: TensorBase): TensorBase =
        tensor.map(::derivative)
}
