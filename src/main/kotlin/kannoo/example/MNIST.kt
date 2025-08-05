package kannoo.example

import kannoo.old.Computer
import kannoo.old.Layer
import kannoo.old.Learner
import kannoo.impl.Logistic
import kannoo.impl.MeanSquaredError
import kannoo.old.NeuralNetwork
import kannoo.impl.ReLU
import kannoo.math.Vector
import java.io.FileInputStream
import kotlin.math.round

fun rnd(d: Double): String {
    val r = (round(d * 100.0) / 100.0).toString()
    return r.padEnd(4, ' ')
}

fun targetOf(digit: String): Vector {
    val v = Vector(10)
    v[digit.toInt()] = 1.0
    return v
}

fun inputOf(pixels: List<String>): Vector =
    Vector(pixels.map { it.toDouble() / 255.0 }.toDoubleArray())

fun readCSVs(fileName: String): List<Pair<Vector, Vector>> =
    FileInputStream(fileName)
        .readAllBytes()
        .toString(Charsets.UTF_8)
        .split('\n')
        .filter { !it.isBlank() }
        .map { it.split(',') }
        .map { ex -> inputOf(ex.drop(1)) to targetOf(ex[0]) }

fun MNIST() {
    val trainingFile = "./data/mnist_train.csv"
    val testFile = "./data/mnist_test.csv"

    println("Parsing training set...")
    val trainingSet = readCSVs(trainingFile)

    println("Parsing test set...")
    val testSet = readCSVs(testFile)

    val net = NeuralNetwork(
        layers = listOf(
            Layer(28 * 28),
            Layer(64, ReLU),
            Layer(10, Logistic),
        ),
    )
    val computer = Computer(net)
    val learner = Learner(net, MeanSquaredError)

    (1..100).forEach { n ->
        println("")
        println("================================")
        println("")

        println("Training round $n")
        learner.train(trainingSet, learningRate = 0.1, batchSize = 10)

        println("Calculating mean error")
        val count = MutableList(10) { 0 }
        val costSum = MutableList(10) { 0.0 }
        val outputSum = MutableList(10) { Vector(10) }

        testSet.forEach { (input, target) ->
            val digit = (0..9).first { n -> target[n] == 1.0 }
            val output = computer.compute(input)
            count[digit]++
            outputSum[digit].plusAssign(output)
            costSum[digit] += learner.costFunction.compute(target, output)
        }

        println("Mean error: ${costSum.sum() / testSet.size.toDouble()}")
        (0..9).forEach { digit ->
            println(
                "$digit " +
                        "error: ${rnd(costSum[digit] / count[digit].toDouble())}" +
                        "    " +
                        "mean: ${rnd(outputSum[digit][digit] / count[digit].toDouble())}",
            )
        }
    }
}
