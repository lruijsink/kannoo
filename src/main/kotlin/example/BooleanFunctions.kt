package example

import kannoo.Computer
import kannoo.Learner
import kannoo.Logistic
import kannoo.MeanSquaredError
import kannoo.NeuralNetwork
import kannoo.vectorOf
import kotlin.collections.map
import kotlin.math.round
import kotlin.random.Random

fun booleanFunctionsExample() {
    val net = NeuralNetwork(
        layerSizes = listOf(4, 2, 1),
        activationFunction = Logistic,
        costFunction = MeanSquaredError,
    )
    val learn = Learner(net)
    val computer = Computer(net)

    val trainingData = listOf(
        // 0, 0 = and
        vectorOf(0.0, 0.0, 0.0, 0.0) to vectorOf(0.0),
        vectorOf(0.0, 0.0, 1.0, 0.0) to vectorOf(0.0),
        vectorOf(0.0, 0.0, 0.0, 1.0) to vectorOf(0.0),
        vectorOf(0.0, 0.0, 1.0, 1.0) to vectorOf(1.0),

        // 0, 1 = or
        vectorOf(0.0, 1.0, 0.0, 0.0) to vectorOf(0.0),
        vectorOf(0.0, 1.0, 1.0, 0.0) to vectorOf(1.0),
        vectorOf(0.0, 1.0, 0.0, 1.0) to vectorOf(1.0),
        vectorOf(0.0, 1.0, 1.0, 1.0) to vectorOf(1.0),

        // 1, 0 = xor
        vectorOf(1.0, 0.0, 0.0, 0.0) to vectorOf(0.0),
        vectorOf(1.0, 0.0, 1.0, 0.0) to vectorOf(1.0),
        vectorOf(1.0, 0.0, 0.0, 1.0) to vectorOf(1.0),
        vectorOf(1.0, 0.0, 1.0, 1.0) to vectorOf(0.0),

        // 1, 1 = eq
        vectorOf(1.0, 1.0, 0.0, 0.0) to vectorOf(1.0),
        vectorOf(1.0, 1.0, 1.0, 0.0) to vectorOf(0.0),
        vectorOf(1.0, 1.0, 0.0, 1.0) to vectorOf(0.0),
        vectorOf(1.0, 1.0, 1.0, 1.0) to vectorOf(1.0),
    )

    fun rnd(d: Double): String {
        val r = (round(d * 1000.0) / 1000.0).toString()
        return if (d >= 0.0) ' ' + r.padEnd(5, ' ') else r.padEnd(6, ' ')
    }

    var n = 0
    var e = 1000.0
    while (n < 1000 && e > 0.001) {
        val lr = 0.3 + 0.5 * (Random.nextDouble() * Random.nextDouble())
        repeat(10000) {
            learn.train(trainingData, lr, 4)
        }
        e = trainingData.sumOf { (input, target) -> net.costFunction.cost(target, computer.compute(input)) }
        println(
            "${n.toString().padStart(4, ' ')}: [E =${rnd(e)}] " +
                    trainingData.associate { (t, _) ->
                        t.scalars.map { it.toInt() }.joinToString("") to rnd(computer.compute(t)[0])
                    }
        )
        n++
    }

    println()

    net.layers.forEachIndexed { i, layer ->
        if (i > 0) {
            println("   Bias $i: " + layer.bias.scalars.map(::rnd))
            println("Weights $i: " + net.weights[i].rowVectors.map { it.scalars.map(::rnd) })
        }
    }

    println()
    println()

    trainingData.map { it.first }.chunked(4).forEach { chunk ->
        println(chunk.map { round(computer.compute(it)[0]).toInt() })
    }
}
