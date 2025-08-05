package kannoo.impl

import kannoo.core.ActivationFunction

object LeakyReLU : ActivationFunction {
    override fun compute(x: Double) = if (x <= 0.0) 0.01 * x else x
    override fun derivative(x: Double) = if (x <= 0.0) 0.01 else 1.0
}
