package kannoo.impl

import kannoo.core.ActivationFunction
import kannoo.core.BoundedInnerLayer
import kannoo.core.GradientReceiver
import kannoo.core.InnerLayerInitializer
import kannoo.math.Dimensions
import kannoo.math.Padding
import kannoo.math.Shape
import kannoo.math.Tensor
import kannoo.math.Tensor3
import kannoo.math.Tensor4
import kannoo.math.Vector
import kannoo.math.broadcastPlus
import kannoo.math.convOutputDims
import kannoo.math.convolve
import kannoo.math.convolveTransposed
import kannoo.math.kernelsGradient
import kannoo.math.randomTensor

class ConvolutionLayer(
    val inputDimensions: Dimensions,
    val kernels: Tensor4,
    val bias: Vector,
    val padding: Padding? = null,
    val stride: Dimensions? = null,
    override val activationFunction: ActivationFunction,
) : BoundedInnerLayer<Tensor3, Tensor3>() {

    val outputChannels =
        kernels.size

    val kernelDimensions =
        Dimensions(kernels.shape[2], kernels.shape[3])

    override val outputShape: Shape =
        Shape(outputChannels, convOutputDims(inputDimensions, kernelDimensions, padding, stride).toShape())

    override val learnable: List<Tensor> =
        listOf(kernels, bias)

    override fun preActivation(input: Tensor3): Tensor3 =
        Tensor3(outputChannels) { o -> convolve(input, kernels[o], padding, stride) } broadcastPlus bias

    override fun deltaInput(deltaPreActivation: Tensor3, input: Tensor3): Tensor3 =
        convolveTransposed(kernels, deltaPreActivation, inputDimensions, padding, stride)

    override fun gradients(deltaPreActivation: Tensor3, input: Tensor3, gradient: GradientReceiver) {
        gradient(kernels, kernelsGradient(kernels, deltaPreActivation, input, padding, stride))
        gradient(bias, Vector(outputChannels) { o -> deltaPreActivation[o].sum() })
    }
}

fun convolutionLayer(
    kernelSize: Dimensions,
    outputChannels: Int,
    activationFunction: ActivationFunction,
    padding: Padding? = null,
    stride: Dimensions? = null,
) =
    InnerLayerInitializer { inputShape ->
        if (inputShape.rank != 3)
            throw IllegalArgumentException("Convolution input must be a rank 3 tensor, but got $inputShape")

        ConvolutionLayer(
            inputDimensions = Dimensions(height = inputShape[1], width = inputShape[2]),
            kernels = randomTensor(outputChannels, inputShape[0], kernelSize.height, kernelSize.width),
            bias = Vector(outputChannels),
            padding = padding,
            stride = stride,
            activationFunction = activationFunction,
        )
    }
