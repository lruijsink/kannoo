package kannoo.impl

import kannoo.core.CostFunction
import kannoo.math.Vector
import kannoo.math.square

object MeanSquaredError : CostFunction {
    override fun compute(target: Vector, actual: Vector): Float =
        0.5f * square(actual - target).sum()

    override fun derivative(target: Vector, actual: Vector): Vector =
        actual - target
}
