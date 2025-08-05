package kannoo.example

import kannoo.impl.Logistic
import kannoo.impl.ReLU
import kannoo.impl.DenseLayer
import kannoo.core.InputLayer
import kannoo.core.Model
import kannoo.math.randomVector
import kotlin.system.measureTimeMillis

fun performanceTest2() {
    val inputSize = 28 * 28
    val outputSize = 10
    val model = Model(
        InputLayer(inputSize),
        DenseLayer(10000, ReLU),
        DenseLayer(10000, ReLU),
        DenseLayer(4000, ReLU),
        DenseLayer(1000, ReLU),
        DenseLayer(outputSize, Logistic),
    )

    println("Total parameters: ${model.layers.sumOf { (it as DenseLayer).trainableParameterCount }}")

    val repeats = 20
    val rounds = 10
    repeat(repeats) {
        val inputs = List(rounds) { randomVector(inputSize) }
        val elapsed = measureTimeMillis {
            inputs.forEach { input ->
                model.compute(input)
            }
        }
        println("$elapsed ms")
    }
}
