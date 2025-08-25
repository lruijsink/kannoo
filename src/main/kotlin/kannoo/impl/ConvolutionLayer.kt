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
import kannoo.math.sumOver

class ConvolutionLayer(
    val inputChannels: Int,
    val inputDimensions: Dimensions,
    val kernels: NTensor<NTensor<Matrix>>,
    val bias: Vector,
    val padding: Padding? = null,
    val stride: Dimensions? = null,
    override val activationFunction: ActivationFunction,
) : BoundedInnerLayer<NTensor<Matrix>, NTensor<Matrix>>() {

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
        kernels = NTensor(outputChannels) {
            NTensor(inputChannels) {
                randomMatrix(kernelDimensions.height, kernelDimensions.width)
            }
        },
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

    override fun preActivation(input: NTensor<Matrix>): NTensor<Matrix> =
        NTensor(outputChannels) { o -> convolve(input, kernels[o], padding, stride) } broadcastPlus bias

    override fun deltaInput(deltaPreActivation: NTensor<Matrix>, input: NTensor<Matrix>): NTensor<Matrix> {
        val deltaInput = NTensor(inputChannels) { Matrix(rows = inputDimensions.height, cols = inputDimensions.width) }

        val ph = padding?.height ?: 0
        val pw = padding?.width ?: 0
        val sh = stride?.height ?: 1
        val sw = stride?.width ?: 1

        fun paddedIndex(i: Int, s: Int): Int =
            padding?.scheme?.map(i, s) ?: i

        for (c in 0 until inputChannels) {
            for (o in 0 until outputChannels) {
                for (i in 0 until outputShape[1]) {
                    for (j in 0 until outputShape[2]) {
                        for (m in 0 until kernelDimensions.height) {
                            for (n in 0 until kernelDimensions.width) {
                                val iPad = paddedIndex(i * sh + m - ph, inputDimensions.height)
                                val jPad = paddedIndex(j * sw + n - pw, inputDimensions.width)
                                if (iPad != -1 && jPad != -1)
                                    deltaInput[c][iPad, jPad] += kernels[o][c][m, n] * deltaPreActivation[o][i, j]
                            }
                        }
                    }
                }
            }
        }
        return deltaInput
    }

    override fun gradients(deltaPreActivation: NTensor<Matrix>, input: NTensor<Matrix>, gradient: GradientReceiver) {
        val ph = padding?.height ?: 0
        val pw = padding?.width ?: 0
        val sh = stride?.height ?: 1
        val sw = stride?.width ?: 1

        val kernelGradient = when {
            padding != null && stride != null -> NTensor(outputChannels) { o ->
                NTensor(inputChannels) { c ->
                    Matrix(kernelDimensions.height, kernelDimensions.width) { m, n ->
                        sumOver(0 until outputShape[1], 0 until outputShape[2]) { i, j ->
                            deltaPreActivation[o][i, j] * padding.scheme.pad(i * sh + m - ph, j * sw + n - pw, input[c])
                        }
                    }
                }
            }

            padding != null -> NTensor(outputChannels) { o ->
                NTensor(inputChannels) { c ->
                    Matrix(kernelDimensions.height, kernelDimensions.width) { m, n ->
                        sumOver(0 until outputShape[1], 0 until outputShape[2]) { i, j ->
                            deltaPreActivation[o][i, j] * padding.scheme.pad(i + m - ph, j + n - pw, input[c])
                        }
                    }
                }
            }

            stride != null -> NTensor(outputChannels) { o ->
                NTensor(inputChannels) { c ->
                    Matrix(kernelDimensions.height, kernelDimensions.width) { m, n ->
                        sumOver(0 until outputShape[1], 0 until outputShape[2]) { i, j ->
                            deltaPreActivation[o][i, j] * input[c][i * sh + m, j * sw + n]
                        }
                    }
                }
            }

            else -> NTensor(outputChannels) { o ->
                NTensor(inputChannels) { c ->
                    Matrix(kernelDimensions.height, kernelDimensions.width) { m, n ->
                        sumOver(0 until outputShape[1], 0 until outputShape[2]) { i, j ->
                            deltaPreActivation[o][i, j] * input[c][i + m, j + n]
                        }
                    }
                }
            }
        }

        gradient(kernels, kernelGradient)
        gradient(bias, Vector(outputChannels) { o -> deltaPreActivation[o].sum() })
    }

    private infix fun NTensor<Matrix>.broadcastPlus(vector: Vector): NTensor<Matrix> =
        if (this.size != vector.size) throw IllegalArgumentException("Tensor and vector sizes must match")
        else NTensor(this.size) { i -> this[i].map { it + vector[i] } }
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
