package kannoo.impl

import kannoo.core.ActivationFunction
import kannoo.core.BackPropagation
import kannoo.core.ForwardPass
import kannoo.core.InnerLayer
import kannoo.core.ParameterDelta
import kannoo.core.ParameterDeltas
import kannoo.math.Matrix
import kannoo.math.Vector
import kannoo.math.emptyMatrix
import kannoo.math.outer
import kannoo.math.randomMatrix

class DenseLayer(var weights: Matrix, var bias: Vector, activationFunction: ActivationFunction) :
    InnerLayer(bias.size, activationFunction) {

    constructor(size: Int, activationFunction: ActivationFunction) :
            this(weights = emptyMatrix(), bias = Vector(size) { 0f }, activationFunction = activationFunction)

    val initialized get() = weights.rows > 0

    override fun initialize(previousLayerSize: Int) {
        if (!initialized)
            weights = randomMatrix(size, previousLayerSize)
    }

    override fun computePreActivation(input: Vector): Vector =
        (weights * input) + bias

    override fun backPropagate(forwardPass: ForwardPass, deltaPreActivation: Vector) =
        BackPropagation(
            deltaInput = deltaPreActivation * weights,
            parameterDeltas = ParameterDeltas(
                matrices = listOf(ParameterDelta(weights, outer(deltaPreActivation, forwardPass.input))),
                vectors = listOf(ParameterDelta(bias, deltaPreActivation)),
            ),
        )
}
