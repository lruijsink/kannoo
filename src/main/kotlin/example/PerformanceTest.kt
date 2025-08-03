package example

import kannoo.Computer
import kannoo.Layer
import kannoo.MeanSquaredError
import kannoo.NeuralNetwork
import kannoo.ReLU
import kannoo.randomVector
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
            Layer(outputSize),
        ),
        costFunction = MeanSquaredError,
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
