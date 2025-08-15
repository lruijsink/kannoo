package kannoo.impl

import kannoo.core.ActivationFunction
import kannoo.core.GradientReceiver
import kannoo.core.InnerLayer
import kannoo.math.Matrix
import kannoo.math.Tensor
import kannoo.math.Vector
import kannoo.math.emptyMatrix
import kannoo.math.randomMatrix

class DenseLayer(var weights: Matrix, var bias: Vector, activationFunction: ActivationFunction) :
    InnerLayer(bias.size, activationFunction) {

    constructor(size: Int, activationFunction: ActivationFunction) :
            this(weights = emptyMatrix(), bias = Vector(size), activationFunction = activationFunction)

    override val learnableParameters: List<Tensor<*>> get() = listOf(weights, bias)

    val initialized get() = weights.rows > 0

    override fun initialize(previousLayerSize: Int) {
        if (!initialized)
            weights = randomMatrix(size, previousLayerSize)
    }

    override fun preActivation(input: Vector): Vector =
        weights * input + bias

    override fun deltaInput(deltaPreActivation: Vector, input: Vector): Vector =
        deltaPreActivation * weights

    override fun gradients(deltaPreActivation: Vector, input: Vector, gradient: GradientReceiver) {
        gradient(weights, deltaPreActivation.outer(input))
        gradient(bias, deltaPreActivation)
    }
}
