package kannoo.impl

import kannoo.core.ElementWiseActivationFunction
import kotlin.math.exp

object Logistic : ElementWiseActivationFunction() {
    override fun compute(x: Float) = 1f / (1f + exp(-x))
    override fun derivative(x: Float): Float {
        val f = compute(x)
        return f * (1f - f)
    }
}
