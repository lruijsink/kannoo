package kannoo.core

import kannoo.math.Tensor

fun interface GradientReceiver {
    operator fun invoke(param: Tensor, delta: Tensor)
}
