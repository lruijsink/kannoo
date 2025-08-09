package kannoo.impl

import kannoo.core.ElementWiseActivationFunction

object Linear : ElementWiseActivationFunction() {
    override fun compute(x: Float) = x
    override fun derivative(x: Float) = 1f
}
