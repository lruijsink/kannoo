package kannoo.example

import kannoo.core.InputLayer
import kannoo.core.Model
import kannoo.core.Sample
import kannoo.impl.DenseLayer
import kannoo.impl.Logistic
import kannoo.impl.MeanSquaredError
import kannoo.impl.MiniBatchSGD
import kannoo.impl.ReLU
import kannoo.math.randomVector
import kotlin.system.measureTimeMillis

fun backPropPerformanceTest() {
    val inputLayerSize = 28 * 28
    val hiddenLayerSize = 64
    val hiddenLayerActivationFunction = ReLU
    val outputLayerSize = 10
    val outputLayerActivationFunction = Logistic
    val costFunction = MeanSquaredError
    val learningRate = 0.1f
    val batchSize = 10
    val trainingDataSize = 1000
    val rounds = 10

    val model = Model(
        InputLayer(inputLayerSize),
        DenseLayer(hiddenLayerSize, hiddenLayerActivationFunction),
        DenseLayer(outputLayerSize, outputLayerActivationFunction),
    )
    val sgd = MiniBatchSGD(model, costFunction, batchSize, learningRate)
    val trainingDataNew = List(trainingDataSize) {
        Sample(input = randomVector(inputLayerSize), target = randomVector(outputLayerSize))
    }

    repeat(rounds) { n ->
        println("Round ${n + 1}:")
        val elapsed = measureTimeMillis { sgd.apply(trainingDataNew) }
        println("$elapsed ms")    }
}
