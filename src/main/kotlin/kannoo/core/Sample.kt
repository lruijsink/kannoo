package kannoo.core

import kannoo.math.BoundedTensor

data class Sample<X : BoundedTensor<X>, Y : BoundedTensor<Y>>(
    val input: X,
    val target: Y,
)
