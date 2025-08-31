package kannoo.impl

import kannoo.core.ActivationFunction
import kannoo.core.GradientReceiver
import kannoo.core.InnerLayer
import kannoo.core.InnerLayerInitializer
import kannoo.math.Composite
import kannoo.math.Matrix
import kannoo.math.Shape
import kannoo.math.Tensor
import kannoo.math.Tensor3
import kannoo.math.Tensor4
import kannoo.math.Vector

class FlattenLayer(val inputShape: Shape) : InnerLayer() {

    override val activationFunction: ActivationFunction =
        Linear

    override val learnable: List<Tensor> =
        listOf()

    override val outputShape: Shape =
        Shape(inputShape.totalElements)

    override fun preActivation(input: Tensor): Vector =
        input.flatten()

    override fun preActivationBatch(inputs: Composite): Matrix =
        Matrix(inputs.size) { i -> inputs[i].flatten() }

    override fun deltaInput(deltaPreActivation: Tensor, input: Tensor): Tensor =
        (deltaPreActivation as Vector).unFlatten(inputShape)

    override fun deltaInputBatch(deltaPreActivations: Composite, inputs: Composite): Composite =
        (deltaPreActivations as Matrix).unflattenBatch(inputShape)

    override fun gradients(deltaPreActivation: Tensor, input: Tensor, gradient: GradientReceiver) {
        // Do nothing
    }

    override fun gradientsBatch(deltaPreActivations: Composite, inputs: Composite, gradient: GradientReceiver) {
        // Do nothing
    }

    private fun Matrix.unflattenBatch(shape: Shape): Composite =
        @Suppress("UNCHECKED_CAST")
        when (shape.rank) {
            1 -> throw IllegalArgumentException("Cannot unflatten into a rank 1 shape $shape")
            2 -> Tensor3(size) { i -> this[i].unFlatten(shape) as Matrix }
            3 -> Tensor4(size) { i -> this[i].unFlatten(shape) as Tensor3 }
            else -> TODO("Not yet implemented")
        }
}

fun flattenLayer() = InnerLayerInitializer { inputShape ->
    FlattenLayer(inputShape)
}
