package kannoo.core

import kannoo.math.Vector

abstract class ElementWiseActivationFunction : ActivationFunction {
    abstract fun compute(x: Double): Double
    abstract fun derivative(x: Double): Double

    override fun compute(v: Vector): Vector = v.transform(::compute)
    override fun derivative(v: Vector): Vector = v.transform(::derivative)
}
