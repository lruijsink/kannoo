package kannoo

interface CostFunction {
    fun cost(target: Vector, actual: Vector): Double
    fun costDerivative(target: Vector, actual: Vector): Vector
}

object MeanSquaredError : CostFunction {
    override fun cost(target: Vector, actual: Vector): Double =
        0.5 * square(actual sub target).sum()

    override fun costDerivative(target: Vector, actual: Vector): Vector =
        actual sub target
}
