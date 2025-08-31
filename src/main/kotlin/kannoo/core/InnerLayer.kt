package kannoo.core

import kannoo.math.Composite
import kannoo.math.Shape
import kannoo.math.Tensor

abstract class InnerLayer {

    abstract val outputShape: Shape

    abstract val activationFunction: ActivationFunction

    abstract val learnable: List<Tensor>

    abstract fun preActivation(input: Tensor): Tensor

    abstract fun preActivationBatch(inputs: Composite): Composite

    abstract fun deltaInput(deltaPreActivation: Tensor, input: Tensor): Tensor

    abstract fun deltaInputBatch(deltaPreActivations: Composite, inputs: Composite): Composite

    abstract fun gradients(deltaPreActivation: Tensor, input: Tensor, gradient: GradientReceiver)

    abstract fun gradientsBatch(deltaPreActivations: Composite, inputs: Composite, gradient: GradientReceiver)

    fun compute(input: Tensor): Tensor =
        activationFunction.compute(preActivation(input))

    fun computeBatch(inputs: Composite): Composite =
        activationFunction.compute(preActivationBatch(inputs)) as Composite
}
