package kannoo.core

import kannoo.math.Vector

interface CostFunction {
    fun compute(target: Vector, actual: Vector): Float
    fun derivative(target: Vector, actual: Vector): Vector
}
