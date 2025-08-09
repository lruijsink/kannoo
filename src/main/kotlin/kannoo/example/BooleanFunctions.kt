package kannoo.example

import kannoo.core.InputLayer
import kannoo.core.Model
import kannoo.core.Sample
import kannoo.impl.DenseLayer
import kannoo.impl.Logistic
import kannoo.impl.MeanSquaredError
import kannoo.impl.MiniBatchSGD
import kannoo.math.sumOf
import kannoo.math.vectorOf
import kotlin.math.round

fun booleanFunctionsExample() {
    val cost = MeanSquaredError
    val model = Model(
        InputLayer(4),
        DenseLayer(4, Logistic),
        DenseLayer(1, Logistic),
    )
    val sgd = MiniBatchSGD(model, MeanSquaredError, 4, 0.3f)

    val trainingData = listOf(
        // 0, 0 = and
        Sample(input = vectorOf(0f, 0f, 0f, 0f), target = vectorOf(0f)),
        Sample(input = vectorOf(0f, 0f, 1f, 0f), target = vectorOf(0f)),
        Sample(input = vectorOf(0f, 0f, 0f, 1f), target = vectorOf(0f)),
        Sample(input = vectorOf(0f, 0f, 1f, 1f), target = vectorOf(1f)),

        // 0, 1 = or
        Sample(input = vectorOf(0f, 1f, 0f, 0f), target = vectorOf(0f)),
        Sample(input = vectorOf(0f, 1f, 1f, 0f), target = vectorOf(1f)),
        Sample(input = vectorOf(0f, 1f, 0f, 1f), target = vectorOf(1f)),
        Sample(input = vectorOf(0f, 1f, 1f, 1f), target = vectorOf(1f)),

        // 1, 0 = xor
        Sample(input = vectorOf(1f, 0f, 0f, 0f), target = vectorOf(0f)),
        Sample(input = vectorOf(1f, 0f, 1f, 0f), target = vectorOf(1f)),
        Sample(input = vectorOf(1f, 0f, 0f, 1f), target = vectorOf(1f)),
        Sample(input = vectorOf(1f, 0f, 1f, 1f), target = vectorOf(0f)),

        // 1, 1 = eq
        Sample(input = vectorOf(1f, 1f, 0f, 0f), target = vectorOf(1f)),
        Sample(input = vectorOf(1f, 1f, 1f, 0f), target = vectorOf(0f)),
        Sample(input = vectorOf(1f, 1f, 0f, 1f), target = vectorOf(0f)),
        Sample(input = vectorOf(1f, 1f, 1f, 1f), target = vectorOf(1f)),
    )

    fun rnd(d: Float): String {
        val r = (round(d * 1000f) / 1000f).toString()
        return if (d >= 0f) ' ' + r.padEnd(5, ' ') else r.padEnd(6, ' ')
    }

    var n = 0
    var e = 1000f
    while (n < 1000 && e > 0.01f) {
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
