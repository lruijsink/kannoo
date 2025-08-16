package kannoo.core

import kannoo.math.Tensor

abstract class ElementWiseActivationFunction : ActivationFunction {
    abstract fun compute(x: Float): Float
    abstract fun derivative(x: Float): Float

    final override fun compute(tensor: Tensor): Tensor =
        tensor.map(::compute)

    final override fun derivative(tensor: Tensor): Tensor =
        tensor.map(::derivative)
}
