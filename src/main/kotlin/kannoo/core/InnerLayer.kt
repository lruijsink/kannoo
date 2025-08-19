package kannoo.core

import kannoo.math.BoundedTensor
import kannoo.math.Shape
import kannoo.math.Tensor

abstract class InnerLayer<T : BoundedTensor<T>, O : BoundedTensor<O>>(
    val outputShape: Shape,
    val activationFunction: ActivationFunction,
) {
    abstract val learnable: List<Tensor>

    abstract fun preActivation(input: T): O

    abstract fun deltaInput(deltaPreActivation: O, input: T): T

    abstract fun gradients(deltaPreActivation: O, input: T, gradient: GradientReceiver)

    fun preActivation(input: Tensor): O {
        @Suppress("UNCHECKED_CAST") // TODO: see if this cast can checked with reified dense(...) etc.
        return preActivation(input as T)
    }

    fun deltaInput(deltaPreActivation: Tensor, input: Tensor): T {
        @Suppress("UNCHECKED_CAST") // TODO: see if this cast can checked with reified dense(...) etc.
        return deltaInput(deltaPreActivation as O, input as T)
    }

    fun gradients(deltaPreActivation: Tensor, input: Tensor, gradient: GradientReceiver) {
        @Suppress("UNCHECKED_CAST") // TODO: see if this cast can checked with reified dense(...) etc.
        gradients(deltaPreActivation as O, input as T, gradient)
    }

    fun compute(input: Tensor): O =
        activationFunction.compute(preActivation(input))
}
