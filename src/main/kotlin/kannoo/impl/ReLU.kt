package kannoo.impl

import kannoo.core.ActivationFunction
import kannoo.math.Tensor

object ReLU : ActivationFunction {
    override fun compute(tensor: Tensor): Tensor =
        tensor.map { x -> if (x <= 0f) 0f else x }

    override fun derivative(tensor: Tensor): Tensor =
        tensor.map { x -> if (x <= 0f) 0f else 1f }
}
