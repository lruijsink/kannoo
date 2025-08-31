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
    BoundedInnerLayer<Vector, Vector, Matrix, Matrix>() {

    constructor(inputSize: Int, outputs: Int, activationFunction: ActivationFunction) :
            this(randomMatrix(outputs, inputSize), Vector(outputs), activationFunction)

    override val outputShape: Shape =
        Shape(bias.size)

    override val learnable: List<Tensor> =
        listOf(weights, bias)

    override fun preActivation(input: Vector): Vector =
        weights * input + bias

    override fun preActivationBatch(input: Matrix): Matrix =
        Matrix(input.size) { i -> preActivation(input[i]) }

    override fun deltaInput(deltaPreActivation: Vector, input: Vector): Vector =
        deltaPreActivation * weights

    override fun deltaInputBatch(deltaPreActivation: Matrix, input: Matrix): Matrix =
        Matrix(input.size) { i -> deltaInput(deltaPreActivation[i], input[i]) }

    override fun gradients(deltaPreActivation: Vector, input: Vector, gradient: GradientReceiver) {
        gradient(weights, deltaPreActivation.outer(input))
        gradient(bias, deltaPreActivation)
    }

    override fun gradientsBatch(deltaPreActivation: Matrix, input: Matrix, gradient: GradientReceiver) {
        for (i in 0 until input.size)
            gradients(deltaPreActivation[i], input[i], gradient)
    }
}

fun denseLayer(outputs: Int, activation: ActivationFunction) =
    InnerLayerInitializer { inputShape ->
        if (inputShape.rank > 1)
            throw IllegalArgumentException("Dense layers require a vector (rank 1) input but got $inputShape")

        DenseLayer(inputShape.totalElements, outputs, activation)
    }
