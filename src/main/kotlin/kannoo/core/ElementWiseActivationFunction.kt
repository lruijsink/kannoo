package kannoo.core

import kannoo.math.Vector

abstract class ElementWiseActivationFunction : ActivationFunction {
    abstract fun compute(x: Float): Float
    abstract fun derivative(x: Float): Float

    override fun compute(v: Vector): Vector = v.transform(::compute)
    override fun derivative(v: Vector): Vector = v.transform(::derivative)
}
