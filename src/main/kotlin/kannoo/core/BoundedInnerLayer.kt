package kannoo.core

import kannoo.math.BoundedComposite
import kannoo.math.BoundedTensor
import kannoo.math.Composite
import kannoo.math.Tensor

abstract class BoundedInnerLayer<
        T : BoundedTensor<T>,
        O : BoundedTensor<O>,
        BT : BoundedComposite<BT, T>,
        BO : BoundedComposite<BO, O>,
        > : InnerLayer() {

    abstract fun preActivation(input: T): O

    abstract fun preActivationBatch(inputs: BT): BO

    abstract fun deltaInput(deltaPreActivation: O, input: T): T

    abstract fun deltaInputBatch(deltaPreActivations: BO, inputs: BT): BT

    abstract fun gradients(deltaPreActivation: O, input: T, gradient: GradientReceiver)

    abstract fun gradientsBatch(deltaPreActivations: BO, inputs: BT, gradient: GradientReceiver)

    final override fun preActivation(input: Tensor): O {
        @Suppress("UNCHECKED_CAST") // TODO: see if this cast can checked with reified dense(...) etc.
        return preActivation(input as T)
    }

    final override fun preActivationBatch(inputs: Composite): BO {
        @Suppress("UNCHECKED_CAST")
        return preActivationBatch(inputs as BT)
    }

    final override fun deltaInput(deltaPreActivation: Tensor, input: Tensor): T {
        @Suppress("UNCHECKED_CAST")
        return deltaInput(deltaPreActivation as O, input as T)
    }

    final override fun deltaInputBatch(deltaPreActivations: Composite, inputs: Composite): BT {
        @Suppress("UNCHECKED_CAST")
        return deltaInputBatch(deltaPreActivations as BO, inputs as BT)
    }

    final override fun gradients(deltaPreActivation: Tensor, input: Tensor, gradient: GradientReceiver) {
        @Suppress("UNCHECKED_CAST")
        gradients(deltaPreActivation as O, input as T, gradient)
    }

    final override fun gradientsBatch(deltaPreActivations: Composite, inputs: Composite, gradient: GradientReceiver) {
        @Suppress("UNCHECKED_CAST")
        gradientsBatch(deltaPreActivations as BO, inputs as BT, gradient)
    }
}
