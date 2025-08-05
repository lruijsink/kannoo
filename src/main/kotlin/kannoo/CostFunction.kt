package kannoo

interface CostFunction {
    fun cost(target: Vector, actual: Vector): Double
    fun derivative(target: Vector, actual: Vector): Vector
}

object MeanSquaredError : CostFunction {
    override fun cost(target: Vector, actual: Vector): Double =
        0.5 * (actual - target).square().sum()

    override fun derivative(target: Vector, actual: Vector): Vector =
        actual - target
}
