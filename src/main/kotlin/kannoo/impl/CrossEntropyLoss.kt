package kannoo.impl

import kannoo.core.CostFunction
import kannoo.math.Tensor
import kannoo.math.Vector
import kannoo.math.clip
import kotlin.math.ln

/**
 * NOTE: Only supported with [Softmax] as the output activation layer.
 */
object CrossEntropyLoss : CostFunction {
    const val EPSILON = 1e-15f

    override fun compute(target: Tensor, actual: Tensor): Tensor =
        compute(target as Vector, actual as Vector) // TODO: Implement axis (rows, cols, and for rank > 2 tensors

    override fun derivative(target: Tensor, actual: Tensor): Tensor =
        derivative(target as Vector, actual as Vector) // TODO: Implement axis (rows, cols, and for rank > 2 tensors

    private fun compute(target: Vector, actual: Vector): Vector =
        -target.zip(actual) { t, a -> t * ln(a.clip(EPSILON, 1f - EPSILON)) }

    /**
     * NOTE: This is only correct if combined with [Softmax].
     */
    private fun derivative(target: Vector, actual: Vector): Vector =
        actual - target
}
