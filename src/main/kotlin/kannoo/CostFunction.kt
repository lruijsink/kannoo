package kannoo

interface CostFunction {
    fun cost(target: Vector, actual: Vector): Double
    fun costDerivative(target: Vector, actual: Vector): Vector
}

object MeanSquaredError : CostFunction {
    override fun cost(target: Vector, actual: Vector): Double =
        0.5 * (actual - target).square().sum()

    override fun costDerivative(target: Vector, actual: Vector): Vector =
        actual - target
}
