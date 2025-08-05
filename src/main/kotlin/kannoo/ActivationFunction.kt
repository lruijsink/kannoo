package kannoo

import kotlin.math.exp

interface ActivationFunction {
    fun sigmoid(x: Double): Double
    fun derivative(x: Double): Double
}

fun ActivationFunction.sigmoid(v: Vector): Vector = v.transform(::sigmoid)
fun ActivationFunction.derivative(v: Vector): Vector = v.transform(::derivative)

object Logistic : ActivationFunction {
    override fun sigmoid(x: Double) = 1.0 / (1.0 + exp(-x))
    override fun derivative(x: Double): Double {
        val f = sigmoid(x)
        return f * (1.0 - f)
    }
}

object ReLU : ActivationFunction {
    override fun sigmoid(x: Double) = if (x <= 0.0) 0.0 else x
    override fun derivative(x: Double) = if (x <= 0.0) 0.0 else 1.0
}

object LeakyReLU : ActivationFunction {
    override fun sigmoid(x: Double) = if (x <= 0.0) 0.01 * x else x
    override fun derivative(x: Double) = if (x <= 0.0) 0.01 else 1.0
}

object Linear : ActivationFunction {
    override fun sigmoid(x: Double) = x
    override fun derivative(x: Double) = 1.0
}
