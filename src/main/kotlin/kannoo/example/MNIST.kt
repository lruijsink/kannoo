package kannoo.example

import kannoo.core.CostFunction
import kannoo.core.Model
import kannoo.core.Sample
import kannoo.core.inputLayer
import kannoo.impl.CrossEntropyLoss
import kannoo.impl.GrayscaleConvolutionLayer
import kannoo.impl.Logistic
import kannoo.impl.MeanSquaredError
import kannoo.impl.MiniBatchSGD
import kannoo.impl.Softmax
import kannoo.impl.convolutionLayer
import kannoo.impl.denseLayer
import kannoo.impl.flattenLayer
import kannoo.impl.grayscaleConvolutionLayer
import kannoo.io.readModelFromFile
import kannoo.io.writeMatricesAsRGB
import kannoo.io.writeModelToFile
import kannoo.math.Dimensions
import kannoo.math.Matrix
import kannoo.math.Padding
import kannoo.math.Shape
import kannoo.math.Tensor
import kannoo.math.Tensor3
import kannoo.math.Vector
import kannoo.math.ZeroPadding
import kannoo.math.mean
import java.io.FileInputStream
import java.io.FileOutputStream
import kotlin.math.round
import kotlin.math.roundToInt
import kotlin.system.measureTimeMillis

const val showFullError = true
const val MNIST_MODEL_FILE = "./data/MNIST.kannoo"

fun rnd(d: Float): String {
    val r = (round(d * 100f) / 100f).toString()
    return r.padEnd(4, ' ')
}

fun targetOf(digit: String): Vector {
    val v = Vector(10)
    v[digit.toInt()] = 1f
    return v
}

fun digitOf(target: Tensor): Int =
    (0..9).first { n -> (target as Vector)[n] == 1f }

fun inputOf(pixels: List<String>): Matrix =
    Vector(pixels.map { it.toFloat() / 255f }.toFloatArray()).unFlatten(Shape(28, 28)) as Matrix

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
    val costSum = MutableList(10) { 0f }
    val mseSum = MutableList(10) { 0f }
    val outputSum = MutableList(10) { Vector(10) { 0f } }

    testSet.forEach { (input, target) ->
        val digit = (0..9).first { n -> (target as Vector)[n] == 1f }
        val output = model.compute(input)
        count[digit]++
        outputSum[digit].plusAssign(output)
        costSum[digit] += cost.compute(target, output).mean()
        mseSum[digit] += MeanSquaredError.compute(target, output).mean()
    }

    println(
        "Mean error: " + rnd(costSum.sum() / testSet.size) +
                "  MSE: " + rnd(mseSum.sum() / testSet.size),
    )
    if (!compact) {
        (0..9).forEach { digit ->
            println(
                "   $digit " +
                        "error: ${rnd(costSum[digit] / count[digit].toFloat())}" +
                        "    " +
                        "mean: ${rnd(outputSum[digit][digit] / count[digit].toFloat())}",
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
            inputLayer(28, 28),
            grayscaleConvolutionLayer(
                kernelSize = Dimensions(5, 5),
                outputChannels = 10,
                activationFunction = Logistic,
                padding = Padding(2, 2, ZeroPadding),
                stride = Dimensions(3, 3),
            ),
            convolutionLayer(
                kernelSize = Dimensions(3, 3),
                outputChannels = 16,
                activationFunction = Logistic,
                padding = Padding(1, 1, ZeroPadding),
                stride = Dimensions(2, 2),
            ),
            flattenLayer(),
            denseLayer(36, Logistic),
            denseLayer(10, Softmax),
        )
    }

    val digitSamples = List(10) { i -> i to trainingSet.first { digitOf(it.target) == i }.input }

    (1..100).forEach { n ->
        val sgd = MiniBatchSGD(model = model, cost = cost, batchSize = 64, learningRate = 1f)

        fun Vector.asMatrix(rows: Int, cols: Int): Matrix =
            if (rows * cols != size) throw IllegalArgumentException("Can't convert to that size")
            else Matrix(rows, cols) { i, j -> elements[i * cols + j] }

        //FileOutputStream("./data/MNIST.weights.$n.png").writeMatricesAsRGB(
        //    (model.layers[0] as DenseLayer).weights.rowVectors.map { it.asMatrix(28, 28) },
        //    padding = 2,
        //)

        val subsetSize = 60000

        println()
        println("=====================================")
        println("               ROUND $n")
        println()

        if (showFullError) {
            println("Calculating error over full test set:")
            println()
            showTestSetError(fullTestSet, model, cost)
            println()
        }

        trainingSet.shuffled().chunked(subsetSize).forEachIndexed { i, subSet ->

            println("Training round $n, subset ${i + 1} / ${trainingSet.size / subsetSize}")
            val elapsed = measureTimeMillis {
                sgd.train(subSet)
            }
            println(
                "Took ${rnd(elapsed.toFloat() / 1000)} seconds " +
                        "(${(subSet.size.toFloat() * 1000 / elapsed).roundToInt()} samples per second)"
            )

            writeModelToFile(model, MNIST_MODEL_FILE)
            model.layers.forEachIndexed { i, layer ->
                //FileOutputStream("./data/MNIST.$i.png").writeLayerAsRGB(model.layers[i])
            }

            showTestSetError(miniTestSet, model, cost, true)
            println()
        }

        val L = model.layers[0] as GrayscaleConvolutionLayer
        val M = digitSamples.flatMap { (_, input) ->
            (L.preActivation(input) as Tensor3).slices.toList()
        }

        FileOutputStream("./data/MNIST.$n.convolutions.0.png").writeMatricesAsRGB(M)
    }
}
