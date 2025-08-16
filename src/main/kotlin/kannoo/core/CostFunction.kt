package kannoo.core

import kannoo.math.Tensor
import kannoo.math.TensorBase

abstract class CostFunction {

    fun <T : Tensor<T>> compute(target: T, actual: T): T {
        @Suppress("UNCHECKED_CAST")
        return compute(target as TensorBase, actual as TensorBase) as T
    }

    fun <T : Tensor<T>> derivative(target: T, actual: T): T {
        @Suppress("UNCHECKED_CAST")
        return derivative(target as TensorBase, actual as TensorBase) as T
    }

    abstract fun compute(target: TensorBase, actual: TensorBase): Tensor<*>
    abstract fun derivative(target: TensorBase, actual: TensorBase): Tensor<*>
}
