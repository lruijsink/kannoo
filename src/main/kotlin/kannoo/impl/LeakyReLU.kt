package kannoo.impl

import kannoo.core.ElementWiseActivationFunction

object LeakyReLU : ElementWiseActivationFunction() {
    override fun compute(x: Float) = if (x <= 0f) 0.01f * x else x
    override fun derivative(x: Float) = if (x <= 0f) 0.01f else 1f
}
