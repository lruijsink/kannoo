package kannoo.math

import kotlin.math.abs

fun randomIntVector(size: Int, range: IntRange = -10..10): Vector =
    Vector(size) { range.random().toFloat() }

fun randomUIntVector(size: Int, range: IntRange = 0..10): Vector =
    randomIntVector(size, range)

fun randomIntMatrix(rows: Int, cols: Int, range: IntRange = -10..10): Matrix =
    Matrix(rows, cols) { _, _ -> range.random().toFloat() }

fun randomUIntMatrix(rows: Int, cols: Int, range: IntRange = 0..10): Matrix =
    randomIntMatrix(rows, cols, range)

fun randomIntTensor3(s1: Int, s2: Int, s3: Int, range: IntRange = -10..10): Tensor3 =
    Tensor3(s1, s2, s3) { _, _, _ -> range.random().toFloat() }

fun randomUIntTensor3(s1: Int, s2: Int, s3: Int, range: IntRange = 0..10): Tensor3 =
    randomIntTensor3(s1, s2, s3, range)

fun <T : Tensor> assertTensorEquals(
    expected: T,
    actual: T,
    tolerance: Float = 0.001f,
) {
    expected.zip(actual) { a, b ->
        if (abs(a - b) > tolerance) assertTensorEquals(expected, actual) // fails
        0f
    }
}
