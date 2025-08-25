package kannoo.impl

import kannoo.core.ActivationFunction
import kannoo.core.BoundedInnerLayer
import kannoo.core.GradientReceiver
import kannoo.core.InnerLayerInitializer
import kannoo.math.Dimensions
import kannoo.math.Matrix
import kannoo.math.NTensor
import kannoo.math.Padding
import kannoo.math.Shape
import kannoo.math.Tensor
import kannoo.math.Vector
import kannoo.math.convolutionOutputDimensions
import kannoo.math.convolveGS
import kannoo.math.convolveTransposedGS
import kannoo.math.kernelsGradientGS
import kannoo.math.randomTensor

class GrayscaleConvolutionLayer(
    val inputDimensions: Dimensions,
    val kernels: NTensor<Matrix>,
    val bias: Vector,
    val padding: Padding? = null,
    val stride: Dimensions? = null,
    override val activationFunction: ActivationFunction,
) : BoundedInnerLayer<Matrix, NTensor<Matrix>>() {

    constructor(
        inputDimensions: Dimensions,
        kernelDimensions: Dimensions,
        padding: Padding? = null,
        stride: Dimensions? = null,
        outputChannels: Int,
        activationFunction: ActivationFunction,
    ) : this(
        inputDimensions = inputDimensions,
        kernels = randomTensor(outputChannels, kernelDimensions.height, kernelDimensions.width),
        bias = Vector(outputChannels),
        padding = padding,
        stride = stride,
        activationFunction = activationFunction,
    )

    val outputChannels = kernels.size

    val kernelDimensions = kernels[0].dimensions

    override val outputShape: Shape =
        Shape(
            outputChannels,
            convolutionOutputDimensions(inputDimensions, kernelDimensions, padding, stride).toShape(),
        )

    override val learnable: List<Tensor> =
        listOf(kernels, bias)

    override fun preActivation(input: Matrix): NTensor<Matrix> =
        NTensor(outputChannels) { o -> convolveGS(input, kernels[o], padding, stride) } broadcastPlus bias

    override fun deltaInput(deltaPreActivation: NTensor<Matrix>, input: Matrix): Matrix =
        convolveTransposedGS(kernels, deltaPreActivation, inputDimensions, padding, stride)

    override fun gradients(deltaPreActivation: NTensor<Matrix>, input: Matrix, gradient: GradientReceiver) {
        gradient(kernels, kernelsGradientGS(kernels, deltaPreActivation, input, padding, stride))
        gradient(bias, Vector(outputChannels) { o -> deltaPreActivation[o].sum() })
    }

    // TODO: define generally
    private infix fun NTensor<Matrix>.broadcastPlus(vector: Vector): NTensor<Matrix> =
        if (this.size != vector.size) throw IllegalArgumentException("Tensor and vector sizes must match")
        else NTensor(this.size) { i -> this[i].map { it + vector[i] } }
}

fun grayscaleConvolutionLayer(
    kernelSize: Dimensions,
    outputChannels: Int,
    activationFunction: ActivationFunction,
    padding: Padding? = null,
    stride: Dimensions? = null,
) =
    InnerLayerInitializer { inputShape ->
        if (inputShape.rank != 2)
            throw IllegalArgumentException("Grayscale input must be a matrix (with 2 dimensions) but got $inputShape")

        GrayscaleConvolutionLayer(
            inputDimensions = Dimensions(height = inputShape[0], width = inputShape[1]),
            kernelDimensions = kernelSize,
            padding = padding,
            stride = stride,
            outputChannels = outputChannels,
            activationFunction = activationFunction,
        )
    }
