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
import kannoo.math.convolutionOutputDimensions
import kannoo.math.convolve
import kannoo.math.convolveTransposed
import kannoo.math.kernelsGradient
import kannoo.math.randomTensor

class ConvolutionLayer(
    val inputChannels: Int,
    val inputDimensions: Dimensions,
    val kernels: Tensor4,
    val bias: Vector,
    val padding: Padding? = null,
    val stride: Dimensions? = null,
    override val activationFunction: ActivationFunction,
) : BoundedInnerLayer<Tensor3, Tensor3>() {

    constructor(
        inputChannels: Int,
        inputDimensions: Dimensions,
        kernelDimensions: Dimensions,
        padding: Padding? = null,
        stride: Dimensions? = null,
        outputChannels: Int,
        activationFunction: ActivationFunction,
    ) : this(
        inputChannels = inputChannels,
        inputDimensions = inputDimensions,
        kernels = randomTensor(outputChannels, inputChannels, kernelDimensions.height, kernelDimensions.width),
        bias = Vector(outputChannels),
        padding = padding,
        stride = stride,
        activationFunction = activationFunction,
    )

    val outputChannels = kernels.size

    val kernelDimensions = Dimensions(kernels.shape[2], kernels.shape[3])

    override val outputShape: Shape =
        Shape(
            outputChannels,
            convolutionOutputDimensions(inputDimensions, kernelDimensions, padding, stride).toShape(),
        )

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

    // TODO: define generally
    private infix fun Tensor3.broadcastPlus(vector: Vector): Tensor3 =
        if (this.size != vector.size) throw IllegalArgumentException("Tensor and vector sizes must match")
        else Tensor3(this.size) { i -> this[i].map { it + vector[i] } }
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
            inputChannels = inputShape[0],
            inputDimensions = Dimensions(height = inputShape[1], width = inputShape[2]),
            kernelDimensions = kernelSize,
            padding = padding,
            stride = stride,
            outputChannels = outputChannels,
            activationFunction = activationFunction,
        )
    }
