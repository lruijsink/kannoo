package kannoo.math

import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

fun randomFloat(): Float =
    2f * Random.nextFloat() - 1f

fun Float.clip(minValue: Float, maxValue: Float) =
    max(minValue, min(this, maxValue))

fun <T> Iterable<T>.sumOf(fn: (T) -> Float): Float {
    var acc = 0f
    for (el in this) acc += fn(el)
    return acc
}
