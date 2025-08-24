package kannoo.impl

import kannoo.core.ActivationFunction
import kannoo.math.Tensor

object LeakyReLU : ActivationFunction {
    override fun compute(tensor: Tensor): Tensor =
        tensor.map { x -> if (x <= 0f) 0.01f * x else x }

    override fun derivative(tensor: Tensor): Tensor =
        tensor.map { x -> if (x <= 0f) 0.01f else 1f }
}
