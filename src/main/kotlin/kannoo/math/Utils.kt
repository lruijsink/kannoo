package kannoo.math

import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * @return Random float between `-1f` and `1f`
 */
fun randomSignedFloat(): Float =
    2f * Random.nextFloat() - 1f

/**
 * @param Size vector to generate
 *
 * @return Vector of size [size] with all elements initialized to [randomSignedFloat]`()`
 */
fun randomVector(size: Int): Vector =
    Vector(size) { randomSignedFloat() }

/**
 * @param rows Row count to generate
 *
 * @param cols Column count to generate
 *
 * @return `rows` x `cols` matrix with all elements initialized to [randomSignedFloat]`()`
 */
fun randomMatrix(rows: Int, cols: Int): Matrix =
    Matrix(rows = rows, cols = cols) { _, _ -> randomSignedFloat() }

/**
 * Clips `this` to a given range, also referred to as 'clamping'
 *
 * @param minValue Min value to clip to
 *
 * @param maxValue Max value to clip to
 */
fun Float.clip(minValue: Float, maxValue: Float) =
    max(minValue, min(this, maxValue))

/**
 * Overload of `sumOf` for summing `Float`, this is presumably absent from the standard library as it is prone to
 * overflow into NaN in certain cases.
 *
 * @param function Function to apply over this [Iterable] to get the float values to sum
 *
 * @return Summed values of applying [function] to `this`
 */
fun <T> Iterable<T>.sumOf(function: (T) -> Float): Float {
    var acc = 0f
    for (el in this) acc += function(el)
    return acc
}

/**
 * Convenience function, computes the square of [x]
 *
 * @param x The number to square
 *
 * @return x * x
 */
fun square(x: Float): Float =
    x * x

/**
 * Get the Euclidean norm (2-norm / square norm) over a
 */
fun euclideanNorm(tensors: Iterable<Tensor>): Float {
    val maxElement = tensors.maxOf { it.max() }
    var scaledSumSquared = 0f
    for (tensor in tensors)
        tensor.forEachElement { scaledSumSquared += square(it / maxElement) }
    return maxElement * sqrt(scaledSumSquared)
}

/**
 * @return Mean average value over all elements in the tensor
 */
fun Tensor.mean(): Float =
    sum() / totalElements

/**
 * @param iRange Range of `i` values
 *
 * @param jRange Range of `j` values
 *
 * @param compute Function to compute
 *
 * @return Sum of [compute] for each (i, j) in range
 */
inline fun sumOver(iRange: IntRange, jRange: IntRange, crossinline compute: (i: Int, j: Int) -> Float): Float {
    var res = 0f
    for (i in iRange)
        for (j in jRange)
            res += compute(i, j)
    return res
}

/**
 * @param iRange Range of `i` values
 *
 * @param jRange Range of `j` values
 *
 * @param kRange Range of `k` values
 *
 * @param compute Function to compute
 *
 * @return Sum of [compute] for each (i, j, k) in range
 */
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

/**
 * Sum over a pair of indexes starting at 0 and ending at `xMax` (excl.)
 *
 * @param iMax Max `i` value (excl.)
 *
 * @param jMax Max `j` value (excl.)
 *
 * @param compute Function to compute
 *
 * @return Sum of [compute] for each (i, j) in `[0, iMax)`, `[0, jMax)`
 */
inline fun sumTo(iMax: Int, jMax: Int, crossinline compute: (i: Int, j: Int) -> Float) =
    sumOver(0 until iMax, 0 until jMax, compute)

/**
 * Sum over a triple of indexes starting at 0 and ending at `xMax` (excl.)
 *
 * @param iMax Max `i` value (excl.)
 *
 * @param jMax Max `j` value (excl.)
 *
 * @param kMax Max `k` value (excl.)
 *
 * @param compute Function to compute
 *
 * @return Sum of [compute] for each (i, j, k) in `[0, iMax)`, `[0, jMax)`, `[0, kMax)`
 */
inline fun sumTo(iMax: Int, jMax: Int, kMax: Int, crossinline compute: (i: Int, j: Int, k: Int) -> Float) =
    sumOver(0 until iMax, 0 until jMax, 0 until kMax, compute)
