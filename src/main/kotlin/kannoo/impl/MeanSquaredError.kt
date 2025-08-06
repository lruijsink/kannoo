package kannoo.impl

import kannoo.core.CostFunction
import kannoo.math.Vector

object MeanSquaredError : CostFunction {
    override fun compute(target: Vector, actual: Vector): Double =
        0.5 * (actual - target).square().sum()

    override fun derivative(target: Vector, actual: Vector): Vector =
        actual - target
}
