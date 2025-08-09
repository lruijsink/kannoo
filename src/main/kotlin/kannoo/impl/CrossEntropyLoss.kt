package kannoo.impl

import kannoo.core.CostFunction
import kannoo.math.Vector
import kannoo.math.clip
import kannoo.math.zipSumOf
import kotlin.math.ln

/**
 * NOTE: Only supported with [Softmax] as the output activation layer.
 */
object CrossEntropyLoss : CostFunction {
    const val EPSILON = 1e-15f

    override fun compute(target: Vector, actual: Vector): Float =
        -target.zipSumOf(actual) { t, a -> t * ln(a.clip(EPSILON, 1f - EPSILON)) }

    /**
     * NOTE: This is only correct if combined with [Softmax].
     */
    override fun derivative(target: Vector, actual: Vector): Vector =
        actual - target
}
