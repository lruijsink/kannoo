package kannoo.impl

import kannoo.core.ActivationFunction
import kannoo.core.BoundedInnerLayer
import kannoo.core.GradientReceiver
import kannoo.core.InnerLayerInitializer
import kannoo.math.Matrix
import kannoo.math.Shape
import kannoo.math.Tensor
import kannoo.math.Vector
import kannoo.math.randomMatrix

class DenseLayer(val weights: Matrix, val bias: Vector, override val activationFunction: ActivationFunction) :
    BoundedInnerLayer<Vector, Vector>() {

    constructor(inputSize: Int, outputs: Int, activationFunction: ActivationFunction) :
            this(randomMatrix(outputs, inputSize), Vector(outputs), activationFunction)

    override val outputShape: Shape =
        Shape(bias.size)

    override val learnable: List<Tensor> =
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
        if (inputShape.dimensions.size > 1)
            throw IllegalArgumentException("Dense layers require have 1-dimensional input")

        DenseLayer(inputShape.totalElements, outputs, activation)
    }
