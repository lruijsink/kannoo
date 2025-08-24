package kannoo.core

import kannoo.math.Tensor

data class Sample(
    val input: Tensor,
    val target: Tensor,
)
