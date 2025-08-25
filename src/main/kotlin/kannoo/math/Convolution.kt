package kannoo.math

fun convolve(
    input: Tensor3,
    kernels: Tensor3,
    padding: Padding? = null,
    stride: Dimensions? = null,
): Matrix =
    when {
        padding != null && stride != null -> convolveImpl(input, kernels, padding, stride)
        padding != null -> convolveImpl(input, kernels, padding)
        stride != null -> convolveImpl(input, kernels, stride)
        else -> convolveImpl(input, kernels)
    }

fun convolutionOutputDimensions(
    input: Dimensions,
    kernel: Dimensions,
    padding: Padding? = null,
    stride: Dimensions? = null,
): Dimensions =
    Dimensions(
        height = (input.height + 2 * (padding?.height ?: 0) - kernel.height) / (stride?.height ?: 1) + 1,
        width = (input.width + 2 * (padding?.width ?: 0) - kernel.width) / (stride?.width ?: 1) + 1,
    )

private fun convolveImpl(input: Tensor3, kernels: Tensor3, padding: Padding, stride: Dimensions) =
    Matrix(convolutionOutputDimensions(input[0].dimensions, kernels[0].dimensions, padding, stride)) { i, j ->
        sumTo(kernels.shape[0], kernels.shape[1], kernels.shape[2]) { c, m, n ->
            kernels[c][m, n] * padding.scheme.pad(
                i = i * stride.height + m - padding.height,
                j = j * stride.width + n - padding.width,
                input = input[c],
            )
        }
    }

private fun convolveImpl(input: Tensor3, kernels: Tensor3, padding: Padding): Matrix =
    Matrix(convolutionOutputDimensions(input[0].dimensions, kernels[0].dimensions, padding = padding)) { i, j ->
        sumTo(kernels.shape[0], kernels.shape[1], kernels.shape[2]) { c, m, n ->
            kernels[c][m, n] * padding.scheme.pad(i = i + m - padding.height, j = j + n - padding.width, input[c])
        }
    }

private fun convolveImpl(input: Tensor3, kernels: Tensor3, stride: Dimensions): Matrix =
    Matrix(convolutionOutputDimensions(input[0].dimensions, kernels[0].dimensions, stride = stride)) { i, j ->
        sumTo(kernels.shape[0], kernels.shape[1], kernels.shape[2]) { c, m, n ->
            kernels[c][m, n] * input[c][i * stride.height + m, j * stride.width + n]
        }
    }

private fun convolveImpl(input: Tensor3, kernels: Tensor3): Matrix =
    Matrix(convolutionOutputDimensions(input[0].dimensions, kernels[0].dimensions)) { i, j ->
        sumTo(kernels.shape[0], kernels.shape[1], kernels.shape[2]) { c, m, n ->
            kernels[c][m, n] * input[c][i + m, j + n]
        }
    }

fun convolveTransposed(
    kernels: Tensor4,
    deltaPreActivation: Tensor3,
    inputDimensions: Dimensions,
    padding: Padding? = null,
    stride: Dimensions? = null,
): Tensor3 {
    val (outputChannels, inputChannels, kernelHeight, kernelWidth) = kernels.shape
    val (_, outputHeight, outputWidth) = deltaPreActivation.shape
    val deltaInput = Tensor3(inputChannels, inputDimensions.height, inputDimensions.width)

    val ph = padding?.height ?: 0
    val pw = padding?.width ?: 0
    val sh = stride?.height ?: 1
    val sw = stride?.width ?: 1

    fun paddedIndex(i: Int, s: Int): Int =
        padding?.scheme?.map(i, s) ?: i

    for (c in 0 until inputChannels) {
        for (o in 0 until outputChannels) {
            for (i in 0 until outputHeight) {
                for (j in 0 until outputWidth) {
                    for (m in 0 until kernelHeight) {
                        for (n in 0 until kernelWidth) {
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

fun kernelsGradient(
    kernels: Tensor4,
    deltaPreActivation: Tensor3,
    input: Tensor3,
    padding: Padding?,
    stride: Dimensions?,
): Tensor4 {
    val (outputChannels, inputChannels, kernelHeight, kernelWidth) = kernels.shape
    val (_, outputHeight, outputWidth) = deltaPreActivation.shape

    val ph = padding?.height ?: 0
    val pw = padding?.width ?: 0
    val sh = stride?.height ?: 1
    val sw = stride?.width ?: 1

    return when {
        padding != null && stride != null ->
            Tensor4(outputChannels, inputChannels, kernelHeight, kernelWidth) { o, c, m, n ->
                sumTo(outputHeight, outputWidth) { i, j ->
                    deltaPreActivation[o][i, j] * padding.scheme.pad(i * sh + m - ph, j * sw + n - pw, input[c])
                }
            }

        padding != null ->
            Tensor4(outputChannels, inputChannels, kernelHeight, kernelWidth) { o, c, m, n ->
                sumTo(outputHeight, outputWidth) { i, j ->
                    deltaPreActivation[o][i, j] * padding.scheme.pad(i + m - ph, j + n - pw, input[c])
                }
            }

        stride != null ->
            Tensor4(outputChannels, inputChannels, kernelHeight, kernelWidth) { o, c, m, n ->
                sumTo(outputHeight, outputWidth) { i, j ->
                    deltaPreActivation[o][i, j] * input[c][i * sh + m, j * sw + n]
                }
            }

        else ->
            Tensor4(outputChannels, inputChannels, kernelHeight, kernelWidth) { o, c, m, n ->
                sumTo(outputHeight, outputWidth) { i, j ->
                    deltaPreActivation[o][i, j] * input[c][i + m, j + n]
                }
            }
    }
}
