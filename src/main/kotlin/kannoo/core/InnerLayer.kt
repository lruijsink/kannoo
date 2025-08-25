package kannoo.core

import kannoo.math.Shape
import kannoo.math.Tensor

abstract class InnerLayer {

    abstract val outputShape: Shape

    abstract val activationFunction: ActivationFunction

    abstract val learnable: List<Tensor>

    abstract fun preActivation(input: Tensor): Tensor

    abstract fun deltaInput(deltaPreActivation: Tensor, input: Tensor): Tensor

    abstract fun gradients(deltaPreActivation: Tensor, input: Tensor, gradient: GradientReceiver)

    fun compute(input: Tensor): Tensor =
        activationFunction.compute(preActivation(input))
}
