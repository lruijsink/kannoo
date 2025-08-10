package kannoo.math

import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

// TODO: doc
fun randomSignedFloat(): Float =
    2f * Random.nextFloat() - 1f

// TODO: doc
fun randomVector(size: Int): Vector =
    Vector(size) { randomSignedFloat() }

// TODO: doc
fun randomMatrix(rows: Int, cols: Int): Matrix =
    Matrix(rows = rows, cols = cols) { _, _ -> randomSignedFloat() }

// TODO: doc
fun Float.clip(minValue: Float, maxValue: Float) =
    max(minValue, min(this, maxValue))

// TODO: doc
fun <T> Iterable<T>.sumOf(function: (T) -> Float): Float {
    var acc = 0f
    for (el in this) acc += function(el)
    return acc
}
