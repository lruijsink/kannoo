package kannoo.core

import kannoo.math.BoundedTensor
import kannoo.math.Tensor

abstract class BoundedInnerLayer<T : BoundedTensor<T>, O : BoundedTensor<O>> : InnerLayer() {

    abstract fun preActivation(input: T): O

    abstract fun deltaInput(deltaPreActivation: O, input: T): T

    abstract fun gradients(deltaPreActivation: O, input: T, gradient: GradientReceiver)

    override fun preActivation(input: Tensor): O {
        @Suppress("UNCHECKED_CAST") // TODO: see if this cast can checked with reified dense(...) etc.
        return preActivation(input as T)
    }

    override fun deltaInput(deltaPreActivation: Tensor, input: Tensor): T {
        @Suppress("UNCHECKED_CAST") // TODO: see if this cast can checked with reified dense(...) etc.
        return deltaInput(deltaPreActivation as O, input as T)
    }

    override fun gradients(deltaPreActivation: Tensor, input: Tensor, gradient: GradientReceiver) {
        @Suppress("UNCHECKED_CAST") // TODO: see if this cast can checked with reified dense(...) etc.
        gradients(deltaPreActivation as O, input as T, gradient)
    }
}
