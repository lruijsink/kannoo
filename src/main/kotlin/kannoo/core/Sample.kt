package kannoo.core

import kannoo.math.BoundedTensor

data class Sample<T : BoundedTensor<T>>(
    val input: T,
    val target: T,
)
