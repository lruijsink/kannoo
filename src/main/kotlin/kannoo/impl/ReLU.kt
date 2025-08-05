package kannoo.impl

import kannoo.core.ActivationFunction

object ReLU : ActivationFunction {
    override fun compute(x: Double) = if (x <= 0.0) 0.0 else x
    override fun derivative(x: Double) = if (x <= 0.0) 0.0 else 1.0
}
