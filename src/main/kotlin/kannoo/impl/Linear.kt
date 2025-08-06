package kannoo.impl

import kannoo.core.ElementWiseActivationFunction

object Linear : ElementWiseActivationFunction() {
    override fun compute(x: Double) = x
    override fun derivative(x: Double) = 1.0
}
