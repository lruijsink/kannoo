package kannoo.core

import kannoo.math.Vector

abstract class InnerLayer(
    val size: Int,
    val activationFunction: ActivationFunction,
) {
    abstract fun initialize(previousLayerSize: Int)

    abstract fun backPropagate(forwardPass: ForwardPass, deltaPreActivation: Vector): BackPropagation

    protected abstract fun computePreActivation(input: Vector): Vector

    fun forwardPass(input: Vector): ForwardPass {
        val preActivation = computePreActivation(input)
        return ForwardPass(input, preActivation, activationFunction.compute(preActivation))
    }
}
