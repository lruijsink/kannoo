package kannoo.math

import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

fun randomDouble() =
    2.0 * Random.nextDouble() - 1.0

fun Double.clip(minValue: Double, maxValue: Double) =
    max(minValue, min(this, maxValue))
