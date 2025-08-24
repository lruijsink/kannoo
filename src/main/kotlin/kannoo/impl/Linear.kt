package kannoo.impl

import kannoo.core.ActivationFunction
import kannoo.math.Tensor

object Linear : ActivationFunction {
    override fun compute(tensor: Tensor): Tensor =
        tensor

    override fun derivative(tensor: Tensor): Tensor =
        tensor.map { 1f }
}
