package example

import kannoo.Layer
import kannoo.Learner
import kannoo.Logistic
import kannoo.MeanSquaredError
import kannoo.NeuralNetwork
import kannoo.ReLU
import kannoo.api.DenseLayer
import kannoo.api.InputLayer
import kannoo.api.MiniBatchSGD
import kannoo.api.Model
import kannoo.api.TrainingExample
import kannoo.randomVector
import kotlin.system.measureTimeMillis

private const val inputLayerSize = 28 * 28
private const val hiddenLayerSize = 64
private val hiddenLayerActivationFunction = ReLU
private const val outputLayerSize = 10
private val outputLayerActivationFunction = Logistic
private val costFunction = MeanSquaredError
private const val learningRate = 0.1
private const val batchSize = 10
private const val trainingDataSize = 1000
private const val rounds = 10

fun performanceTest3() {
    /**
     * New:
     */
    val model = Model(
        InputLayer(inputLayerSize),
        DenseLayer(hiddenLayerSize, hiddenLayerActivationFunction),
        DenseLayer(outputLayerSize, outputLayerActivationFunction),
    )
    val sgd = MiniBatchSGD(model, costFunction, batchSize, learningRate)
    val trainingDataNew = List(trainingDataSize) {
        TrainingExample(input = randomVector(inputLayerSize), target = randomVector(outputLayerSize))
    }

    /**
     * Old:
     */
    val net = NeuralNetwork(
        layers = listOf(
            Layer(inputLayerSize),
            Layer(hiddenLayerSize, hiddenLayerActivationFunction),
            Layer(outputLayerSize, outputLayerActivationFunction),
        ),
    )
    val learner = Learner(net, costFunction)
    val trainingDataOld = List(trainingDataSize) {
        Pair(randomVector(inputLayerSize), randomVector(outputLayerSize))
    }

    /**
     * Test:
     */
    repeat(rounds) { n ->
        println("Round ${n + 1}:")

        val elapsedNew = measureTimeMillis { sgd.apply(trainingDataNew) }
        println("New: $elapsedNew ms")

        val elapsedOld = measureTimeMillis { learner.train(trainingDataOld, learningRate, batchSize) }
        println("Old: $elapsedOld ms")

        println()
    }
}
