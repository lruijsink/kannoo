package kannoo.impl

import kannoo.core.ActivationFunction

object Linear : ActivationFunction {
    override fun compute(x: Double) = x
    override fun derivative(x: Double) = 1.0
}
