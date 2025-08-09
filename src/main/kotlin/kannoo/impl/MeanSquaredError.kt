package kannoo.impl

import kannoo.core.CostFunction
import kannoo.math.Vector

object MeanSquaredError : CostFunction {
    override fun compute(target: Vector, actual: Vector): Float =
        0.5f * (actual - target).square().sum()

    override fun derivative(target: Vector, actual: Vector): Vector =
        actual - target
}
