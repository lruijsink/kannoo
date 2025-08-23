package kannoo.impl

import kannoo.core.ActivationFunction
import kannoo.core.GradientReceiver
import kannoo.core.InnerLayer
import kannoo.core.InnerLayerInitializer
import kannoo.math.Shape
import kannoo.math.Tensor
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

    override fun deltaInput(deltaPreActivation: Tensor, input: Tensor): Tensor =
        (deltaPreActivation as Vector).unFlatten(inputShape)

    override fun gradients(deltaPreActivation: Tensor, input: Tensor, gradient: GradientReceiver) {
        // Do nothing
    }
}

fun flattenLayer() = InnerLayerInitializer { inputShape ->
    FlattenLayer(inputShape)
}
