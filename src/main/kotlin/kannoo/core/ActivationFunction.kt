package kannoo.core

import kannoo.math.BoundedTensor
import kannoo.math.Tensor

interface ActivationFunction {
    fun compute(tensor: Tensor): Tensor
    fun derivative(tensor: Tensor): Tensor

    fun <T : BoundedTensor<T>> compute(tensor: T): T {
        @Suppress("UNCHECKED_CAST")
        return compute(tensor as Tensor) as T
    }

    fun <T : BoundedTensor<T>> derivative(tensor: T): T {
        @Suppress("UNCHECKED_CAST")
        return derivative(tensor as Tensor) as T
    }
}
