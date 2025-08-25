package kannoo.math

fun convolve(input: Matrix, kernel: Matrix, padding: Padding? = null, stride: Dimensions? = null): Matrix =
    when {
        padding != null && stride != null -> convolveImpl(input, kernel, padding, stride)
        padding != null -> convolveImpl(input, kernel, padding)
        stride != null -> convolveImpl(input, kernel, stride)
        else -> convolveImpl(input, kernel)
    }

fun convolve(
    input: NTensor<Matrix>,
    kernels: NTensor<Matrix>,
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

inline fun sumOver(iRange: IntRange, jRange: IntRange, crossinline compute: (i: Int, j: Int) -> Float): Float {
    var res = 0f
    for (i in iRange)
        for (j in jRange)
            res += compute(i, j)
    return res
}

inline fun sumOver(
    iRange: IntRange,
    jRange: IntRange,
    kRange: IntRange,
    crossinline compute: (i: Int, j: Int, k: Int) -> Float,
): Float {
    var res = 0f
    for (i in iRange)
        for (j in jRange)
            for (k in kRange)
                res += compute(i, j, k)
    return res
}

private inline fun Matrix(dimensions: Dimensions, crossinline init: (i: Int, j: Int) -> Float): Matrix =
    Matrix(rows = dimensions.height, cols = dimensions.width, init = init)

private fun convolveImpl(input: Matrix, kernel: Matrix, padding: Padding, stride: Dimensions): Matrix =
    Matrix(convolutionOutputDimensions(input.dimensions, kernel.dimensions, padding, stride)) { i, j ->
        sumOver(0 until kernel.rows, 0 until kernel.cols) { u, v ->
            kernel[u, v] * padding.scheme.pad(
                i = i * stride.height + u - padding.height,
                j = j * stride.width + v - padding.width,
                input = input,
            )
        }
    }

private fun convolveImpl(input: Matrix, kernel: Matrix, padding: Padding): Matrix =
    Matrix(convolutionOutputDimensions(input.dimensions, kernel.dimensions, padding = padding)) { i, j ->
        sumOver(0 until kernel.rows, 0 until kernel.cols) { u, v ->
            kernel[u, v] * padding.scheme.pad(i = i + u - padding.height, j = j + v - padding.width, input)
        }
    }

private fun convolveImpl(input: Matrix, kernel: Matrix, stride: Dimensions): Matrix =
    Matrix(convolutionOutputDimensions(input.dimensions, kernel.dimensions, stride = stride)) { i, j ->
        sumOver(0 until kernel.rows, 0 until kernel.cols) { u, v ->
            kernel[u, v] * input[i * stride.height + u, j * stride.width + v]
        }
    }

private fun convolveImpl(input: Matrix, kernel: Matrix): Matrix =
    Matrix(convolutionOutputDimensions(input.dimensions, kernel.dimensions)) { i, j ->
        sumOver(0 until kernel.rows, 0 until kernel.cols) { u, v ->
            kernel[u, v] * input[i + u, j + v]
        }
    }

private fun convolveImpl(input: NTensor<Matrix>, kernels: NTensor<Matrix>, padding: Padding, stride: Dimensions) =
    Matrix(convolutionOutputDimensions(input[0].dimensions, kernels[0].dimensions, padding, stride)) { i, j ->
        sumOver(0 until kernels.shape[0], 0 until kernels.shape[1], 0 until kernels.shape[2]) { c, m, n ->
            kernels[c][m, n] * padding.scheme.pad(
                i = i * stride.height + m - padding.height,
                j = j * stride.width + n - padding.width,
                input = input[c],
            )
        }
    }

private fun convolveImpl(input: NTensor<Matrix>, kernels: NTensor<Matrix>, padding: Padding): Matrix =
    Matrix(convolutionOutputDimensions(input[0].dimensions, kernels[0].dimensions, padding = padding)) { i, j ->
        sumOver(0 until kernels.shape[0], 0 until kernels.shape[1], 0 until kernels.shape[2]) { c, m, n ->
            kernels[c][m, n] * padding.scheme.pad(i = i + m - padding.height, j = j + n - padding.width, input[c])
        }
    }

private fun convolveImpl(input: NTensor<Matrix>, kernels: NTensor<Matrix>, stride: Dimensions): Matrix =
    Matrix(convolutionOutputDimensions(input[0].dimensions, kernels[0].dimensions, stride = stride)) { i, j ->
        sumOver(0 until kernels.shape[0], 0 until kernels.shape[1], 0 until kernels.shape[2]) { c, m, n ->
            kernels[c][m, n] * input[c][i * stride.height + m, j * stride.width + n]
        }
    }

private fun convolveImpl(input: NTensor<Matrix>, kernels: NTensor<Matrix>): Matrix =
    Matrix(convolutionOutputDimensions(input[0].dimensions, kernels[0].dimensions)) { i, j ->
        sumOver(0 until kernels.shape[0], 0 until kernels.shape[1], 0 until kernels.shape[2]) { c, m, n ->
            kernels[c][m, n] * input[c][i + m, j + n]
        }
    }
