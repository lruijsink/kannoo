package kannoo.example

import kannoo.core.InputLayer
import kannoo.core.Model
import kannoo.impl.DenseLayer
import kannoo.impl.Logistic
import kannoo.impl.ReLU
import kannoo.math.randomVector
import kotlin.system.measureTimeMillis

fun computePerformanceTest() {
    val inputLayerSize = 28 * 28
    val innerLayerSizes = listOf(10000, 10000, 4000, 1000)
    val innerLayerActivationFunction = ReLU
    val outputLayerSize = 10
    val outputLayerActivationFunction = Logistic
    val rounds = 20

    val model = Model(
        InputLayer(inputLayerSize),
        innerLayerSizes.map { DenseLayer(it, innerLayerActivationFunction) } +
                DenseLayer(outputLayerSize, outputLayerActivationFunction),
    )

    repeat(rounds) {
        val input = randomVector(inputLayerSize)
        val elapsed = measureTimeMillis { model.compute(input) }
        println("$elapsed ms")
    }
}
