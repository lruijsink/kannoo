package kannoo.core

import kannoo.math.Tensor
import kannoo.math.TensorBase

interface ActivationFunction {
    fun compute(tensor: TensorBase): TensorBase
    fun derivative(tensor: TensorBase): TensorBase

    fun <T : Tensor<T>> compute(tensor: T): T {
        @Suppress("UNCHECKED_CAST")
        return compute(tensor as TensorBase) as T
    }

    fun <T : Tensor<T>> derivative(tensor: T): T {
        @Suppress("UNCHECKED_CAST")
        return derivative(tensor as TensorBase) as T
    }
}
