package kannoo.impl

import kannoo.core.ElementWiseActivationFunction

object ReLU : ElementWiseActivationFunction() {
    override fun compute(x: Float) = if (x <= 0f) 0f else x
    override fun derivative(x: Float) = if (x <= 0f) 0f else 1f
}
