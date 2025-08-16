package kannoo.impl

import kannoo.core.CostFunction
import kannoo.math.Tensor
import kannoo.math.square
import kannoo.math.times

object MeanSquaredError : CostFunction {
    override fun compute(target: Tensor, actual: Tensor): Tensor =
        square(actual - target)

    override fun derivative(target: Tensor, actual: Tensor): Tensor =
        2f * (actual - target)
}
