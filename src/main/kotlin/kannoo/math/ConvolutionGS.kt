package kannoo.math

fun convolveGS(input: Matrix, kernel: Matrix, padding: Padding? = null, stride: Dimensions? = null): Matrix =
    when {
        padding != null && stride != null -> convolveImpl(input, kernel, padding, stride)
        padding != null -> convolveImpl(input, kernel, padding)
        stride != null -> convolveImpl(input, kernel, stride)
        else -> convolveImpl(input, kernel)
    }

private fun convolveImpl(input: Matrix, kernel: Matrix, padding: Padding, stride: Dimensions): Matrix =
    Matrix(convolutionOutputDimensions(input.dimensions, kernel.dimensions, padding, stride)) { i, j ->
        sumTo(kernel.rows, kernel.cols) { u, v ->
            kernel[u, v] * padding.scheme.pad(
                i = i * stride.height + u - padding.height,
                j = j * stride.width + v - padding.width,
                input = input,
            )
        }
    }

private fun convolveImpl(input: Matrix, kernel: Matrix, padding: Padding): Matrix =
    Matrix(convolutionOutputDimensions(input.dimensions, kernel.dimensions, padding = padding)) { i, j ->
        sumTo(kernel.rows, kernel.cols) { u, v ->
            kernel[u, v] * padding.scheme.pad(i = i + u - padding.height, j = j + v - padding.width, input)
        }
    }

private fun convolveImpl(input: Matrix, kernel: Matrix, stride: Dimensions): Matrix =
    Matrix(convolutionOutputDimensions(input.dimensions, kernel.dimensions, stride = stride)) { i, j ->
        sumTo(kernel.rows, kernel.cols) { u, v ->
            kernel[u, v] * input[i * stride.height + u, j * stride.width + v]
        }
    }

private fun convolveImpl(input: Matrix, kernel: Matrix): Matrix =
    Matrix(convolutionOutputDimensions(input.dimensions, kernel.dimensions)) { i, j ->
        sumTo(kernel.rows, kernel.cols) { u, v ->
            kernel[u, v] * input[i + u, j + v]
        }
    }

fun convolveTransposedGS(
    kernels: NTensor<Matrix>,
    deltaPreActivation: NTensor<Matrix>,
    inputDimensions: Dimensions,
    padding: Padding? = null,
    stride: Dimensions? = null,
): Matrix {
    val (outputChannels, kernelHeight, kernelWidth) = kernels.shape
    val (_, outputHeight, outputWidth) = deltaPreActivation.shape
    val deltaInput = Matrix(rows = inputDimensions.height, cols = inputDimensions.width)

    val ph = padding?.height ?: 0
    val pw = padding?.width ?: 0
    val sh = stride?.height ?: 1
    val sw = stride?.width ?: 1

    fun paddedIndex(i: Int, s: Int): Int =
        padding?.scheme?.map(i, s) ?: i

    for (o in 0 until outputChannels) {
        for (i in 0 until outputHeight) {
            for (j in 0 until outputWidth) {
                for (m in 0 until kernelHeight) {
                    for (n in 0 until kernelWidth) {
                        val iPad = paddedIndex(i * sh + m - ph, inputDimensions.height)
                        val jPad = paddedIndex(j * sw + n - pw, inputDimensions.width)
                        if (iPad != -1 && jPad != -1)
                            deltaInput[iPad, jPad] += kernels[o][m, n] * deltaPreActivation[o][i, j]
                    }
                }
            }
        }
    }
    return deltaInput
}

fun kernelsGradientGS(
    kernels: NTensor<Matrix>,
    deltaPreActivation: NTensor<Matrix>,
    input: Matrix,
    padding: Padding?,
    stride: Dimensions?,
): NTensor<Matrix> {
    val (outputChannels, kernelHeight, kernelWidth) = kernels.shape
    val (_, outputHeight, outputWidth) = deltaPreActivation.shape

    val ph = padding?.height ?: 0
    val pw = padding?.width ?: 0
    val sh = stride?.height ?: 1
    val sw = stride?.width ?: 1

    return when {
        padding != null && stride != null ->
            NTensor(outputChannels, kernelHeight, kernelWidth) { o, m, n ->
                sumTo(outputHeight, outputWidth) { i, j ->
                    deltaPreActivation[o][i, j] * padding.scheme.pad(i * sh + m - ph, j * sw + n - pw, input)
                }
            }

        padding != null ->
            NTensor(outputChannels, kernelHeight, kernelWidth) { o, m, n ->
                sumTo(outputHeight, outputWidth) { i, j ->
                    deltaPreActivation[o][i, j] * padding.scheme.pad(i + m - ph, j + n - pw, input)
                }
            }

        stride != null ->
            NTensor(outputChannels, kernelHeight, kernelWidth) { o, m, n ->
                sumTo(outputHeight, outputWidth) { i, j ->
                    deltaPreActivation[o][i, j] * input[i * sh + m, j * sw + n]
                }
            }

        else ->
            NTensor(outputChannels, kernelHeight, kernelWidth) { o, m, n ->
                sumTo(outputHeight, outputWidth) { i, j ->
                    deltaPreActivation[o][i, j] * input[i + m, j + n]
                }
            }
    }
}
