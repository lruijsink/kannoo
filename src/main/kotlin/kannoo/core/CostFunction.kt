package kannoo.core

import kannoo.math.BoundedTensor
import kannoo.math.Tensor

interface CostFunction {
    fun compute(target: Tensor, actual: Tensor): Tensor
    fun derivative(target: Tensor, actual: Tensor): Tensor

    fun <T : BoundedTensor<T>> compute(target: T, actual: T): T {
        @Suppress("UNCHECKED_CAST")
        return compute(target as Tensor, actual as Tensor) as T
    }

    fun <T : BoundedTensor<T>> derivative(target: T, actual: T): T {
        @Suppress("UNCHECKED_CAST")
        return derivative(target as Tensor, actual as Tensor) as T
    }
}
