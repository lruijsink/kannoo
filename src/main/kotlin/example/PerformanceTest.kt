package example

import kannoo.old.Computer
import kannoo.old.Layer
import kannoo.impl.Logistic
import kannoo.old.NeuralNetwork
import kannoo.impl.ReLU
import kannoo.math.randomVector
import kotlin.system.measureTimeMillis

fun performanceTest() {
    val inputSize = 28 * 28
    val outputSize = 10
    val net = NeuralNetwork(
        layers = listOf(
            Layer(inputSize),
            Layer(10000, ReLU),
            Layer(10000, ReLU),
            Layer(4000, ReLU),
            Layer(1000, ReLU),
            Layer(outputSize, Logistic),
        ),
    )

    val biases = net.layers.sumOf { it.bias.size }
    val weights = net.weights.sumOf { it.rows * it.cols.toLong() }
    println("Total parameters: ${biases + weights}")

    val repeats = 20
    val rounds = 10
    val computer = Computer(net)
    repeat(repeats) {
        val inputs = List(rounds) { randomVector(inputSize) }
        val elapsed = measureTimeMillis {
            inputs.forEach { input ->
                computer.compute(input)
            }
        }
        println("$elapsed ms")
    }
}
