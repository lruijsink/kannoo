package kannoo.core

import kannoo.math.Shape
import kannoo.math.BoundedTensor
import kannoo.math.Tensor

abstract class InnerLayer<T : BoundedTensor<T>, O : BoundedTensor<O>>(
    val outputShape: Shape,
    val activationFunction: ActivationFunction,
) {
    abstract val learnable: List<Tensor>

    abstract fun preActivation(input: T): O

    abstract fun deltaInput(deltaPreActivation: O, input: T): T

    abstract fun gradients(deltaPreActivation: O, input: T, gradient: GradientReceiver)

    fun preActivation(input: Tensor): O =
        preActivation(input as T) // TODO: cast same way as [Tensor.castUnsafe]

    fun deltaInput(deltaPreActivation: Tensor, input: Tensor): T =
        deltaInput(deltaPreActivation as O, input as T) // TODO: cast same way as [Tensor.castUnsafe]

    fun gradients(deltaPreActivation: Tensor, input: Tensor, gradient: GradientReceiver) {
        gradients(deltaPreActivation as O, input as T, gradient) // TODO: cast same way as [Tensor.castUnsafe]
    }

    fun compute(input: Tensor): O =
        activationFunction.compute(preActivation(input))
}
