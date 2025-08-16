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
fun euclideanNorm(tensors: Iterable<TensorBase>): Float {
    val maxElement = tensors.maxOf { it.max() }
    val scaledSumSquared = tensors.sumOf {
        it.reduce { acc, element ->
            acc + square(element / maxElement)
        }
    }
    return maxElement * sqrt(scaledSumSquared)
}

/**
 * @return Mean average value over all elements in the tensor
 */
fun TensorBase.mean(): Float =
    sum() / totalElements
