package kannoo.core

import kannoo.math.Vector

interface CostFunction {
    fun compute(target: Vector, actual: Vector): Double
    fun derivative(target: Vector, actual: Vector): Vector
}
