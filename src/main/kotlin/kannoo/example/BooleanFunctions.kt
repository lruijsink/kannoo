package kannoo.example

import kannoo.core.InputLayer
import kannoo.core.Model
import kannoo.core.Sample
import kannoo.impl.DenseLayer
import kannoo.impl.Logistic
import kannoo.impl.MeanSquaredError
import kannoo.impl.MiniBatchSGD
import kannoo.math.vectorOf
import kotlin.math.round
import kotlin.random.Random

fun booleanFunctionsExample() {
    val cost = MeanSquaredError
    val model = Model(
        InputLayer(4),
        DenseLayer(2, Logistic),
        DenseLayer(1, Logistic),
    )
    val sgd = MiniBatchSGD(model, MeanSquaredError, 4, 0.3)

    val trainingData = listOf(
        // 0, 0 = and
        Sample(input = vectorOf(0.0, 0.0, 0.0, 0.0), target = vectorOf(0.0)),
        Sample(input = vectorOf(0.0, 0.0, 1.0, 0.0), target = vectorOf(0.0)),
        Sample(input = vectorOf(0.0, 0.0, 0.0, 1.0), target = vectorOf(0.0)),
        Sample(input = vectorOf(0.0, 0.0, 1.0, 1.0), target = vectorOf(1.0)),

        // 0, 1 = or
        Sample(input = vectorOf(0.0, 1.0, 0.0, 0.0), target = vectorOf(0.0)),
        Sample(input = vectorOf(0.0, 1.0, 1.0, 0.0), target = vectorOf(1.0)),
        Sample(input = vectorOf(0.0, 1.0, 0.0, 1.0), target = vectorOf(1.0)),
        Sample(input = vectorOf(0.0, 1.0, 1.0, 1.0), target = vectorOf(1.0)),

        // 1, 0 = xor
        Sample(input = vectorOf(1.0, 0.0, 0.0, 0.0), target = vectorOf(0.0)),
        Sample(input = vectorOf(1.0, 0.0, 1.0, 0.0), target = vectorOf(1.0)),
        Sample(input = vectorOf(1.0, 0.0, 0.0, 1.0), target = vectorOf(1.0)),
        Sample(input = vectorOf(1.0, 0.0, 1.0, 1.0), target = vectorOf(0.0)),

        // 1, 1 = eq
        Sample(input = vectorOf(1.0, 1.0, 0.0, 0.0), target = vectorOf(1.0)),
        Sample(input = vectorOf(1.0, 1.0, 1.0, 0.0), target = vectorOf(0.0)),
        Sample(input = vectorOf(1.0, 1.0, 0.0, 1.0), target = vectorOf(0.0)),
        Sample(input = vectorOf(1.0, 1.0, 1.0, 1.0), target = vectorOf(1.0)),
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
            sgd.apply(trainingData)
        }
        e = trainingData.sumOf { (input, target) -> cost.compute(target, model.compute(input)) }
        println(
            "${n.toString().padStart(4, ' ')}: [E =${rnd(e)}] " +
                    trainingData.associate { (t, _) ->
                        t.scalars.map { it.toInt() }.joinToString("") to rnd(model.compute(t)[0])
                    }
        )
        n++
    }

    println()

    model.layers.forEachIndexed { i, layer ->
        val l = layer as DenseLayer
        println("   Bias $i: " + l.bias.scalars.map(::rnd))
        println("Weights $i: " + l.weights.rowVectors.map { it.scalars.map(::rnd) })
    }

    println()
    println()

    trainingData.map { it.input }.chunked(4).forEach { chunk ->
        println(chunk.map { round(model.compute(it)[0]).toInt() })
    }
}
