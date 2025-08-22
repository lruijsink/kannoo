package kannoo.math

fun convolve(input: Matrix, kernel: Matrix, padding: Padding? = null, stride: Dimensions? = null): Matrix =
    when {
        padding != null && stride != null -> convolveImpl(input, kernel, padding, stride)
        padding != null -> convolveImpl(input, kernel, padding)
        stride != null -> convolveImpl(input, kernel, stride)
        else -> convolveImpl(input, kernel)
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

private inline fun sumOver(iRange: IntRange, jRange: IntRange, crossinline compute: (i: Int, j: Int) -> Float): Float {
    var res = 0f
    for (i in iRange)
        for (j in jRange)
            res += compute(i, j)
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
