package example

import kannoo.impl.Logistic
import kannoo.impl.MeanSquaredError
import kannoo.math.vectorOf
import kannoo.old.Computer
import kannoo.old.Layer
import kannoo.old.Learner
import kannoo.old.NeuralNetwork
import kotlin.math.round
import kotlin.random.Random

fun booleanFunctionsExample() {
    val net = NeuralNetwork(
        layers = listOf(
            Layer(4),
            Layer(2, Logistic),
            Layer(1, Logistic),
        ),
    )
    val learn = Learner(net, MeanSquaredError)
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
        e = trainingData.sumOf { (input, target) -> learn.costFunction.compute(target, computer.compute(input)) }
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
