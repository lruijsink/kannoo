package kannoo.example

import kannoo.core.InputLayer
import kannoo.core.Model
import kannoo.core.Sample
import kannoo.impl.DenseLayer
import kannoo.impl.Logistic
import kannoo.impl.MeanSquaredError
import kannoo.impl.MiniBatchSGD
import kannoo.impl.denseLayer
import kannoo.math.Vector
import kannoo.math.mean
import kannoo.math.sumOf
import kannoo.math.vector
import kotlin.math.round

fun booleanFunctionsExample() {
    val cost = MeanSquaredError
    val model = Model(
        InputLayer(4),
        denseLayer(4, Logistic),
        denseLayer(1, Logistic),
    )
    val sgd = MiniBatchSGD(model, MeanSquaredError, 0.3f, 4)

    val trainingData = listOf(
        // 0, 0 = and
        Sample(input = vector(0f, 0f, 0f, 0f), target = vector(0f)),
        Sample(input = vector(0f, 0f, 1f, 0f), target = vector(0f)),
        Sample(input = vector(0f, 0f, 0f, 1f), target = vector(0f)),
        Sample(input = vector(0f, 0f, 1f, 1f), target = vector(1f)),

        // 0, 1 = or
        Sample(input = vector(0f, 1f, 0f, 0f), target = vector(0f)),
        Sample(input = vector(0f, 1f, 1f, 0f), target = vector(1f)),
        Sample(input = vector(0f, 1f, 0f, 1f), target = vector(1f)),
        Sample(input = vector(0f, 1f, 1f, 1f), target = vector(1f)),

        // 1, 0 = xor
        Sample(input = vector(1f, 0f, 0f, 0f), target = vector(0f)),
        Sample(input = vector(1f, 0f, 1f, 0f), target = vector(1f)),
        Sample(input = vector(1f, 0f, 0f, 1f), target = vector(1f)),
        Sample(input = vector(1f, 0f, 1f, 1f), target = vector(0f)),

        // 1, 1 = eq
        Sample(input = vector(1f, 1f, 0f, 0f), target = vector(1f)),
        Sample(input = vector(1f, 1f, 1f, 0f), target = vector(0f)),
        Sample(input = vector(1f, 1f, 0f, 1f), target = vector(0f)),
        Sample(input = vector(1f, 1f, 1f, 1f), target = vector(1f)),
    )

    fun rnd(d: Float): String {
        val r = (round(d * 1000f) / 1000f).toString()
        return if (d >= 0f) ' ' + r.padEnd(5, ' ') else r.padEnd(6, ' ')
    }

    var n = 0
    var e = 1000f
    while (n < 1000 && e > 0.01f) {
        repeat(10000) {
            sgd.train(trainingData)
        }
        e = trainingData.sumOf { (input, target) -> cost.compute(target, model.compute(input)).mean() }
        println(
            "${n.toString().padStart(4, ' ')}: [E =${rnd(e)}] " +
                    trainingData.associate { (t, _) ->
                        (t as Vector).elements.map { it.toInt() }.joinToString("") to rnd((model.compute(t) as Vector)[0])
                    }
        )
        n++
    }

    println()

    model.layers.forEachIndexed { i, layer ->
        val l = layer as DenseLayer
        println("   Bias $i: " + l.bias.elements.map(::rnd))
        println("Weights $i: " + l.weights.rowVectors.map { it.elements.map(::rnd) })
    }

    println()
    println()

    trainingData.map { it.input }.chunked(4).forEach { chunk ->
        println(chunk.map { round((model.compute(it) as Vector)[0]).toInt() })
    }
}
