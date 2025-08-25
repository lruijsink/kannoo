package kannoo.impl

import kannoo.core.ActivationFunction
import kannoo.math.Tensor
import kotlin.math.exp

object Logistic : ActivationFunction {
    override fun compute(tensor: Tensor): Tensor =
        tensor.map { x -> 1f / (1f + exp(-x)) }

    override fun derivative(tensor: Tensor): Tensor =
        tensor.map { x ->
            val f = 1f / (1f + exp(-x))
            f * (1f - f)
        }
}
