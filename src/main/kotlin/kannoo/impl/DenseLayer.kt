package kannoo.impl

import kannoo.core.ActivationFunction
import kannoo.core.BackPropagation
import kannoo.core.ForwardPass
import kannoo.core.InnerLayer
import kannoo.core.ParameterDelta
import kannoo.core.ParameterDeltas
import kannoo.core.compute
import kannoo.core.derivative
import kannoo.math.Vector
import kannoo.math.emptyMatrix
import kannoo.math.hadamard
import kannoo.math.outer
import kannoo.math.randomMatrix
import kannoo.math.transposeDot

class DenseLayer(
    size: Int,
    activationFunction: ActivationFunction,
) : InnerLayer(size, activationFunction) {

    var weights = emptyMatrix()
    val bias = Vector(size)

    val trainableParameterCount
        get() = weights.rows * weights.rows + bias.size

    override fun initialize(previousLayerSize: Int) {
        weights = randomMatrix(size, previousLayerSize)
    }

    override fun forwardPass(input: Vector): ForwardPass {
        if (input.size != weights.cols) throw IllegalArgumentException("Expecting input of size $size")
        val preActivation = (weights * input) + bias
        return ForwardPass(
            input = input,
            preActivation = preActivation,
            output = activationFunction.compute(preActivation),
        )
    }

    override fun backPropagate(forwardPass: ForwardPass, deltaOutput: Vector): BackPropagation {
        val deltaPreActivation = hadamard(deltaOutput, activationFunction.derivative(forwardPass.preActivation))
        return BackPropagation(
            deltaInput = transposeDot(weights, deltaPreActivation),
            parameterDeltas = ParameterDeltas(
                matrices = listOf(ParameterDelta(weights, outer(deltaPreActivation, forwardPass.input))),
                vectors = listOf(ParameterDelta(bias, deltaPreActivation)),
            )
        )
    }
}