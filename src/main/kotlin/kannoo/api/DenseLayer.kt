package kannoo.api

import kannoo.ActivationFunction
import kannoo.Vector
import kannoo.derivative
import kannoo.emptyMatrix
import kannoo.hadamard
import kannoo.outer
import kannoo.randomMatrix
import kannoo.sigmoid
import kannoo.transposeDot

class DenseLayer(
    size: Int,
    activationFunction: ActivationFunction,
) : InnerLayer(size, activationFunction) {

    private var weights = emptyMatrix()
    private val bias = Vector(size)

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
            output = activationFunction.sigmoid(preActivation),
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
