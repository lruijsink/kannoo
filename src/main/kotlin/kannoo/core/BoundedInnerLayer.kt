package kannoo.core

import kannoo.math.BoundedTensor
import kannoo.math.Composite
import kannoo.math.Tensor

abstract class BoundedInnerLayer<
        T : BoundedTensor<T>,
        O : BoundedTensor<O>,
        BT : Composite<BT, T>,
        BO : Composite<BO, O>,
        > : InnerLayer() {

    abstract fun preActivation(input: T): O

    abstract fun preActivationBatch(input: BT): BO

    abstract fun deltaInput(deltaPreActivation: O, input: T): T

    abstract fun deltaInputBatch(deltaPreActivation: BO, input: BT): BT

    abstract fun gradients(deltaPreActivation: O, input: T, gradient: GradientReceiver)

    abstract fun gradientsBatch(deltaPreActivation: BO, input: BT, gradient: GradientReceiver)

    final override fun preActivation(input: Tensor): O {
        @Suppress("UNCHECKED_CAST") // TODO: see if this cast can checked with reified dense(...) etc.
        return preActivation(input as T)
    }

    final override fun preActivationBatch(input: Tensor): BO {
        @Suppress("UNCHECKED_CAST")
        return preActivationBatch(input as BT)
    }

    final override fun deltaInput(deltaPreActivation: Tensor, input: Tensor): T {
        @Suppress("UNCHECKED_CAST")
        return deltaInput(deltaPreActivation as O, input as T)
    }

    final override fun deltaInputBatch(deltaPreActivation: Tensor, input: Tensor): BT {
        @Suppress("UNCHECKED_CAST")
        return deltaInputBatch(deltaPreActivation as BO, input as BT)
    }

    final override fun gradients(deltaPreActivation: Tensor, input: Tensor, gradient: GradientReceiver) {
        @Suppress("UNCHECKED_CAST")
        gradients(deltaPreActivation as O, input as T, gradient)
    }

    final override fun gradientsBatch(deltaPreActivation: Tensor, input: Tensor, gradient: GradientReceiver) {
        @Suppress("UNCHECKED_CAST")
        gradientsBatch(deltaPreActivation as BO, input as BT, gradient)
    }
}
