package kannoo.impl

import kannoo.core.ActivationFunction
import kannoo.math.Vector
import kotlin.math.exp

/**
 * NOTE: This is only supported in the output layer, in combination with [CrossEntropyLoss].
 */
object Softmax : ActivationFunction {
    override fun compute(v: Vector): Vector {
        val vMax = v.max()
        val vExp = v.map { exp(it - vMax) }
        val vExpSum = vExp.sum()
        return vExp / vExpSum
    }

    /**
     * Computed by [CrossEntropyLoss] as a combined derivative for efficiency/simplicity.
     */
    override fun derivative(v: Vector): Vector = v
}
