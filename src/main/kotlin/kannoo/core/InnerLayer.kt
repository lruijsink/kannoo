package kannoo.core

import kannoo.math.Tensor
import kannoo.math.Vector

abstract class InnerLayer(
    val size: Int,
    val activationFunction: ActivationFunction,
) {
    abstract val learnableParameters: List<Tensor<*>>

    abstract fun initialize(previousLayerSize: Int)

    abstract fun preActivation(input: Vector): Vector

    abstract fun deltaInput(deltaPreActivation: Vector, input: Vector): Vector

    abstract fun gradients(deltaPreActivation: Vector, input: Vector, gradient: GradientReceiver)

    fun compute(input: Vector): Vector =
        activationFunction.compute(preActivation(input))
}
