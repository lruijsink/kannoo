package kannoo.impl

import kannoo.core.ActivationFunction
import kannoo.core.BoundedInnerLayer
import kannoo.core.GradientReceiver
import kannoo.core.InnerLayerInitializer
import kannoo.math.Matrix
import kannoo.math.Shape
import kannoo.math.Tensor
import kannoo.math.Vector
import kannoo.math.broadcastPlus
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

    override fun preActivationBatch(inputs: Matrix): Matrix =
        inputs.multiplyTranspose(weights).broadcastPlus(bias)

    override fun deltaInput(deltaPreActivation: Vector, input: Vector): Vector =
        deltaPreActivation * weights

    override fun deltaInputBatch(deltaPreActivations: Matrix, inputs: Matrix): Matrix =
        deltaPreActivations * weights

    override fun gradients(deltaPreActivation: Vector, input: Vector, gradient: GradientReceiver) {
        gradient(weights, deltaPreActivation.outer(input))
        gradient(bias, deltaPreActivation)
    }

    override fun gradientsBatch(deltaPreActivations: Matrix, inputs: Matrix, gradient: GradientReceiver) {
        gradient(weights, deltaPreActivations.transposeMultiply(inputs))
        gradient(bias, deltaPreActivations.sumRows())
    }
}

fun denseLayer(outputs: Int, activation: ActivationFunction) =
    InnerLayerInitializer { inputShape ->
        if (inputShape.rank > 1)
            throw IllegalArgumentException("Dense layers require a vector (rank 1) input but got $inputShape")

        DenseLayer(inputShape.totalElements, outputs, activation)
    }
