package kannoo.core

import kannoo.math.Vector

abstract class InnerLayer(
    val size: Int,
    val activationFunction: ActivationFunction,
) {
    abstract fun initialize(previousLayerSize: Int)

    abstract fun forwardPass(input: Vector): ForwardPass

    abstract fun backPropagate(forwardPass: ForwardPass, deltaOutput: Vector, skipDeltaInput: Boolean): BackPropagation
}
