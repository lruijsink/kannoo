package kannoo.impl

import kannoo.core.ActivationFunction
import kannoo.core.GradientReceiver
import kannoo.core.InnerLayer
import kannoo.math.Matrix
import kannoo.math.NTensor
import kannoo.math.Shape
import kannoo.math.Tensor

class GrayscaleConvolutionLayer(
    val inputRows: Int,
    val inputCols: Int,
    val outputChannels: Int,
    activationFunction: ActivationFunction,
) :
    InnerLayer<Matrix, NTensor<Matrix>>(
        outputShape = Shape(outputChannels, inputRows, inputCols),
        activationFunction = activationFunction
    ) {

    override val learnable: List<Tensor<*>>
        get() = TODO("Not yet implemented")

    override fun preActivation(input: Matrix): NTensor<Matrix> {
        TODO("Not yet implemented")
    }

    override fun deltaInput(deltaPreActivation: NTensor<Matrix>, input: Matrix): Matrix {
        TODO("Not yet implemented")
    }

    override fun gradients(deltaPreActivation: NTensor<Matrix>, input: Matrix, gradient: GradientReceiver) {
        TODO("Not yet implemented")
    }
}
