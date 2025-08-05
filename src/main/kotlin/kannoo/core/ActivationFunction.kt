package kannoo.core

import kannoo.math.Vector

interface ActivationFunction {
    fun compute(x: Double): Double
    fun derivative(x: Double): Double
}

fun ActivationFunction.compute(v: Vector): Vector = v.transform(::compute)
fun ActivationFunction.derivative(v: Vector): Vector = v.transform(::derivative)
