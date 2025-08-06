package kannoo.impl

import kannoo.core.ElementWiseActivationFunction
import kotlin.math.exp

object Logistic : ElementWiseActivationFunction() {
    override fun compute(x: Double) = 1.0 / (1.0 + exp(-x))
    override fun derivative(x: Double): Double {
        val f = compute(x)
        return f * (1.0 - f)
    }
}
