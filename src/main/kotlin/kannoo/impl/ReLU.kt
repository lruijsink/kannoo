package kannoo.impl

import kannoo.core.ElementWiseActivationFunction

object ReLU : ElementWiseActivationFunction() {
    override fun compute(x: Double) = if (x <= 0.0) 0.0 else x
    override fun derivative(x: Double) = if (x <= 0.0) 0.0 else 1.0
}
