package kannoo.impl

import kannoo.core.ActivationFunction
import kannoo.core.BackPropagation
import kannoo.core.ForwardPass
import kannoo.core.InnerLayer
import kannoo.core.ParameterDelta
import kannoo.core.ParameterDeltas
import kannoo.math.Vector
import kannoo.math.emptyMatrix
import kannoo.math.hadamard
import kannoo.math.outer
import kannoo.math.randomMatrix
import kannoo.math.times

class DenseLayer(
    size: Int,
    activationFunction: ActivationFunction,
) : InnerLayer(size, activationFunction) {

    var weights = emptyMatrix()
    var bias = Vector(size)

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
        val deltaPreActivation =
            if (activationFunction is Softmax) deltaOutput
            else hadamard(deltaOutput, activationFunction.derivative(forwardPass.preActivation))

        return BackPropagation(
            deltaInput = deltaPreActivation * weights,
            parameterDeltas = ParameterDeltas(
                matrices = listOf(ParameterDelta(weights, outer(deltaPreActivation, forwardPass.input))),
                vectors = listOf(ParameterDelta(bias, deltaPreActivation)),
            )
        )
    }
}
