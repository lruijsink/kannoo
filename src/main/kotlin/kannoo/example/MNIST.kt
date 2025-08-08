package kannoo.example

import kannoo.core.CostFunction
import kannoo.core.InputLayer
import kannoo.core.Model
import kannoo.core.Sample
import kannoo.impl.CrossEntropyLoss
import kannoo.impl.DenseLayer
import kannoo.impl.MeanSquaredError
import kannoo.impl.MiniBatchSGD
import kannoo.impl.ReLU
import kannoo.impl.Softmax
import kannoo.io.readModelFromFile
import kannoo.io.writeModelToFile
import kannoo.math.Vector
import java.io.FileInputStream
import kotlin.math.round

const val MNIST_MODEL_FILE = "./data/MNIST.kannoo"

fun rnd(d: Double): String {
    val r = (round(d * 100.0) / 100.0).toString()
    return r.padEnd(4, ' ')
}

fun rnd5(d: Double): String {
    val r = (round(d * 1000.0) / 1000.0).toString()
    return r.padEnd(5, ' ')
}

fun targetOf(digit: String): Vector {
    val v = Vector(10)
    v[digit.toInt()] = 1.0
    return v
}

fun inputOf(pixels: List<String>): Vector =
    Vector(pixels.map { it.toDouble() / 255.0 }.toDoubleArray())

fun readCSVs(fileName: String): List<Sample> =
    FileInputStream(fileName)
        .readAllBytes()
        .toString(Charsets.UTF_8)
        .split('\n')
        .filter { !it.isBlank() }
        .map { it.split(',') }
        .map { ex -> Sample(input = inputOf(ex.drop(1)), target = targetOf(ex[0])) }

fun showTestSetError(testSet: List<Sample>, model: Model, cost: CostFunction, compact: Boolean = false) {
    val count = MutableList(10) { 0 }
    val costSum = MutableList(10) { 0.0 }
    val mseSum = MutableList(10) { 0.0 }
    val outputSum = MutableList(10) { Vector(10) }

    testSet.forEach { (input, target) ->
        val digit = (0..9).first { n -> target[n] == 1.0 }
        val output = model.compute(input)
        count[digit]++
        outputSum[digit].plusAssign(output)
        costSum[digit] += cost.compute(target, output)
        mseSum[digit] += MeanSquaredError.compute(target, output)
    }

    println(
        "Mean error: " + rnd(costSum.sum() / testSet.size) +
                "  MSE: " + rnd(mseSum.sum() / testSet.size),
    )
    if (!compact) {
        (0..9).forEach { digit ->
            println(
                "   $digit " +
                        "error: ${rnd(costSum[digit] / count[digit].toDouble())}" +
                        "    " +
                        "mean: ${rnd(outputSum[digit][digit] / count[digit].toDouble())}",
            )
        }
    }
}

fun MNIST() {
    val trainingFile = "./data/mnist_train.csv"
    val testFile = "./data/mnist_test.csv"

    println("Parsing training set...")
    val trainingSet = readCSVs(trainingFile)

    println("Parsing test set...")
    val fullTestSet = readCSVs(testFile)
    val miniTestSet = fullTestSet.shuffled().take(1000)

    val cost = CrossEntropyLoss
    val model = try {
        readModelFromFile(MNIST_MODEL_FILE)
            .also { println("Successfully loaded pre-trained model") }
    } catch (e: Exception) {
        println("Pre-trained model not found, creating new instance ($e)")
        Model(
            InputLayer(28 * 28),
            DenseLayer(256, ReLU),
            DenseLayer(64, ReLU),
            DenseLayer(10, Softmax),
        )
    }
    val sgd = MiniBatchSGD(model, cost, 64, 0.1)

    (1..100).forEach { n ->
        val subsetSize = 6000

        println()
        println("=====================================")
        println("               ROUND $n")
        println()
        println("Calculating error over full test set:")
        println()
        showTestSetError(fullTestSet, model, cost)
        println()

        trainingSet.shuffled().chunked(subsetSize).forEachIndexed { i, subSet ->

            println("Training round $n, subset ${i + 1} / ${trainingSet.size / subsetSize}")
            sgd.apply(subSet)
            writeModelToFile(model, MNIST_MODEL_FILE)

            showTestSetError(miniTestSet, model, cost, true)
            println()
        }
    }
}
