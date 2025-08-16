package kannoo.core

import kannoo.math.Tensor

data class Sample<T : Tensor<T>>(
    val input: T,
    val target: T,
)
