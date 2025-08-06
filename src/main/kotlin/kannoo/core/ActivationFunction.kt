package kannoo.core

import kannoo.math.Vector

interface ActivationFunction {
    fun compute(v: Vector): Vector
    fun derivative(v: Vector): Vector
}
