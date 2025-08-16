package kannoo.impl

import kannoo.core.CostFunction
import kannoo.math.Tensor
import kannoo.math.TensorBase
import kannoo.math.square
import kannoo.math.times

object MeanSquaredError : CostFunction() {
    override fun compute(target: TensorBase, actual: TensorBase): Tensor<*> =
        square((actual as Tensor<*>) - (target as Tensor<*>)) // TODO: Move generic methods into TensorBase

    override fun derivative(target: TensorBase, actual: TensorBase): Tensor<*> =
        2f * ((actual as Tensor<*>) - (target as Tensor<*>)) // TODO: Move generic methods into TensorBase
}
