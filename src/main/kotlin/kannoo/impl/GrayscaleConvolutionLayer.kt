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
import kannoo.math.convolve
import kannoo.math.randomMatrix

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
        kernels = NTensor(outputChannels) { randomMatrix(kernelDimensions.height, kernelDimensions.width) },
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
            convolutionOutputDimensions(inputDimensions, kernels[0].dimensions, padding, stride).toShape(),
        )

    override val learnable: List<Tensor> =
        listOf(kernels, bias)

    override fun preActivation(input: Matrix): NTensor<Matrix> =
        NTensor(outputChannels) { o -> convolve(input, kernels[o], padding, stride) } broadcastPlus bias

    override fun deltaInput(deltaPreActivation: NTensor<Matrix>, input: Matrix): Matrix {
        val deltaInput = Matrix(rows = input.rows, cols = input.cols)
        val ph = padding?.height ?: 0
        val pw = padding?.width ?: 0
        val sh = stride?.height ?: 1
        val sw = stride?.width ?: 1
        for (o in 0 until outputChannels) {
            for (i in 0 until input.rows) {
                for (j in 0 until input.cols) {
                    for (u in 0 until kernelDimensions.height) {
                        for (v in 0 until kernelDimensions.width) {
                            val ia = i + ph - u
                            if (ia < 0 || ia % sh > 0) continue

                            val ja = j + pw - v
                            if (ja < 0 || ja % sw > 0) continue

                            val id = ia / sh
                            val jd = ja / sw
                            if (id < deltaPreActivation[o].rows && jd < deltaPreActivation[o].cols)
                                deltaInput[i, j] += kernels[o][u, v] * deltaPreActivation[o][id, jd]
                        }
                    }
                }
            }
        }
        return deltaInput
    }

    override fun gradients(deltaPreActivation: NTensor<Matrix>, input: Matrix, gradient: GradientReceiver) {
        val kernelGradient = NTensor(outputChannels) {
            Matrix(kernelDimensions.height, kernelDimensions.width)
        }
        val ph = padding?.height ?: 0
        val pw = padding?.width ?: 0
        val sh = stride?.height ?: 1
        val sw = stride?.width ?: 1
        for (c in 0 until outputChannels) {
            for (u in 0 until kernelDimensions.height) {
                for (v in 0 until kernelDimensions.width) {
                    for (i in 0 until outputShape.dimensions[0]) {
                        for (j in 0 until outputShape.dimensions[1]) {
                            val ic = i * sh + u
                            val jc = j * sw + v
                            val x = padding?.scheme?.pad(ic - ph, jc - pw, input) ?: input[ic, jc]
                            kernelGradient[c][u, v] += deltaPreActivation[c][i, j] * x
                        }
                    }
                }
            }
        }

        val biasGradient = Vector(outputChannels) { o -> deltaPreActivation[o].sum() }

        gradient(kernels, kernelGradient)
        gradient(bias, biasGradient)
    }

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
        if (inputShape.dimensions.size != 2)
            throw IllegalArgumentException("Grayscale input must be a matrix (with 2 dimensions) but got $inputShape")

        GrayscaleConvolutionLayer(
            inputDimensions = Dimensions(height = inputShape.dimensions[0], width = inputShape.dimensions[1]),
            kernelDimensions = kernelSize,
            padding = padding,
            stride = stride,
            outputChannels = outputChannels,
            activationFunction = activationFunction,
        )
    }
