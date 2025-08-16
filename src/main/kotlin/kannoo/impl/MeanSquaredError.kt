package kannoo.impl

import kannoo.core.CostFunction
import kannoo.math.TensorBase
import kannoo.math.square
import kannoo.math.times

object MeanSquaredError : CostFunction {
    override fun compute(target: TensorBase, actual: TensorBase): TensorBase =
        square(actual - target)

    override fun derivative(target: TensorBase, actual: TensorBase): TensorBase =
        2f * (actual - target)
}
