package kannoo.impl

import kannoo.core.ActivationFunction
import kannoo.math.Matrix
import kannoo.math.Tensor
import kannoo.math.Vector
import kotlin.math.exp

/**
 * NOTE: This is only supported in the output layer, in combination with [CrossEntropyLoss].
 */
object Softmax : ActivationFunction {
    private fun compute(v: Vector): Vector {
        val vMax = v.max()
        val vExp = v.map { exp(it - vMax) }
        val vExpSum = vExp.sum()
        return vExp / vExpSum
    }

    /**
     * Computed by [CrossEntropyLoss] as a combined derivative for efficiency/simplicity.
     */
    private fun derivative(v: Vector): Vector = v

    override fun compute(tensor: Tensor): Tensor =
        when (tensor) {
            is Vector -> compute(tensor)
            is Matrix -> Matrix(tensor.size) { row -> compute(tensor[row]) }
            else -> TODO("Implement axis (rows, cols, and for rank > 2 tensors")
        }

    override fun derivative(tensor: Tensor): Tensor =
        when (tensor) {
            is Vector -> derivative(tensor)
            is Matrix -> Matrix(tensor.size) { row -> derivative(tensor[row]) }
            else -> TODO("Implement axis (rows, cols, and for rank > 2 tensors")
        }
}
