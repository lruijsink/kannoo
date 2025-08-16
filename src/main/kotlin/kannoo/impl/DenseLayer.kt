package kannoo.impl

import kannoo.core.ActivationFunction
import kannoo.core.GradientReceiver
import kannoo.core.InnerLayer
import kannoo.core.InnerLayerInitializer
import kannoo.math.Matrix
import kannoo.math.Shape
import kannoo.math.Vector
import kannoo.math.randomMatrix

class DenseLayer(val weights: Matrix, val bias: Vector, activationFunction: ActivationFunction) :
    InnerLayer<Vector, Vector>(
        outputShape = Shape(bias.size),
        activationFunction = activationFunction,
    ) {

    constructor(inputShape: Shape, outputs: Int, activationFunction: ActivationFunction) :
            this(randomMatrix(outputs, inputShape.totalElements), Vector(outputs), activationFunction)

    override val learnable =
        listOf(weights, bias)

    override fun preActivation(input: Vector): Vector =
        weights * input + bias

    override fun deltaInput(deltaPreActivation: Vector, input: Vector): Vector =
        deltaPreActivation * weights

    override fun gradients(deltaPreActivation: Vector, input: Vector, gradient: GradientReceiver) {
        gradient(weights, deltaPreActivation.outer(input))
        gradient(bias, deltaPreActivation)
    }
}

fun denseLayer(outputs: Int, activation: ActivationFunction) =
    InnerLayerInitializer { inputShape ->
        DenseLayer(inputShape, outputs, activation)
    }
