package kannoo.core

import kannoo.math.Tensor
import kannoo.math.TensorBase

abstract class ElementWiseActivationFunction : ActivationFunction {
    abstract fun compute(x: Float): Float
    abstract fun derivative(x: Float): Float

    final override fun compute(tensor: TensorBase): Tensor<*> =
        (tensor as Tensor<*>).map(::compute) // TODO: Move generic methods to TensorBase, maybe

    final override fun derivative(tensor: TensorBase): Tensor<*> =
        (tensor as Tensor<*>).map(::derivative) // TODO: Move generic methods to TensorBase, maybe
}
