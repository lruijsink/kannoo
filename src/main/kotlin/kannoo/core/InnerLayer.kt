package kannoo.core

import kannoo.math.Shape
import kannoo.math.Tensor
import kannoo.math.TensorBase

abstract class InnerLayer<T : Tensor<T>, O : Tensor<O>>(
    val outputShape: Shape,
    val activationFunction: ActivationFunction,
) {
    abstract val learnable: List<TensorBase>

    abstract fun preActivation(input: T): O

    abstract fun deltaInput(deltaPreActivation: O, input: T): T

    abstract fun gradients(deltaPreActivation: O, input: T, gradient: GradientReceiver)

    fun preActivation(input: TensorBase): O =
        preActivation(input as T) // TODO: cast same way as [Tensor.castUnsafe]

    fun deltaInput(deltaPreActivation: TensorBase, input: TensorBase): T =
        deltaInput(deltaPreActivation as O, input as T) // TODO: cast same way as [Tensor.castUnsafe]

    fun gradients(deltaPreActivation: TensorBase, input: TensorBase, gradient: GradientReceiver) {
        gradients(deltaPreActivation as O, input as T, gradient) // TODO: cast same way as [Tensor.castUnsafe]
    }

    fun compute(input: TensorBase): O =
        activationFunction.compute(preActivation(input))
}
