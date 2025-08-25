package kannoo.math

fun convolveGS(input: Matrix, kernel: Matrix, padding: Padding?, stride: Dimensions?): Matrix =
    Matrix(convOutputDims(input.dimensions, kernel.dimensions, padding, stride)) { i, j ->
        sumTo(kernel.rows, kernel.cols) { u, v ->
            when {
                padding != null && stride != null ->
                    kernel[u, v] * padding.scheme.pad(
                        i = i * stride.height + u - padding.height,
                        j = j * stride.width + v - padding.width,
                        input = input,
                    )

                padding != null ->
                    kernel[u, v] * padding.scheme.pad(i = i + u - padding.height, j = j + v - padding.width, input)

                stride != null ->
                    kernel[u, v] * input[i * stride.height + u, j * stride.width + v]

                else ->
                    kernel[u, v] * input[i + u, j + v]
            }
        }
    }

fun convolveTransposedGS(
    kernels: Tensor3,
    deltaPreActivation: Tensor3,
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
    kernels: Tensor3,
    deltaPreActivation: Tensor3,
    input: Matrix,
    padding: Padding?,
    stride: Dimensions?,
): Tensor3 {
    val (outputChannels, kernelHeight, kernelWidth) = kernels.shape
    val (_, outputHeight, outputWidth) = deltaPreActivation.shape

    val ph = padding?.height ?: 0
    val pw = padding?.width ?: 0
    val sh = stride?.height ?: 1
    val sw = stride?.width ?: 1

    return Tensor3(outputChannels, kernelHeight, kernelWidth) { o, m, n ->
        sumTo(outputHeight, outputWidth) { i, j ->
            when {
                padding != null && stride != null ->
                    deltaPreActivation[o][i, j] * padding.scheme.pad(i * sh + m - ph, j * sw + n - pw, input)

                padding != null ->
                    deltaPreActivation[o][i, j] * padding.scheme.pad(i + m - ph, j + n - pw, input)

                stride != null ->
                    deltaPreActivation[o][i, j] * input[i * sh + m, j * sw + n]

                else ->
                    deltaPreActivation[o][i, j] * input[i + m, j + n]
            }
        }
    }
}
