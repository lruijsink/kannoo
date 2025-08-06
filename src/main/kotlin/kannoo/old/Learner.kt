package kannoo.old

import kannoo.core.CostFunction
import kannoo.math.Matrix
import kannoo.math.Vector
import kannoo.math.emptyMatrix
import kannoo.math.hadamard
import kannoo.math.outer
import kannoo.math.transposeDot

class Learner(
    val neuralNetwork: NeuralNetwork,
    val costFunction: CostFunction,
) {
    class Batch(neuralNetwork: NeuralNetwork) {
        val deltaBiases = MutableList(neuralNetwork.layers.size) { i ->
            Vector(neuralNetwork.layers[i].size)
        }
        val deltaWeights = MutableList(neuralNetwork.layers.size) { i ->
            if (i == 0) emptyMatrix()
            else Matrix(neuralNetwork.layers[i].size, neuralNetwork.layers[i - 1].size) { 0.0 }
        }
    }

    private val computer = Computer(neuralNetwork)
    private val numLayers = neuralNetwork.layers.size

    fun train(trainingData: List<Pair<Vector, Vector>>, learningRate: Double, batchSize: Int) {
        trainingData.shuffled().chunked(batchSize).forEach { batch(it, learningRate) }
    }

    private fun batch(trainingData: List<Pair<Vector, Vector>>, learningRate: Double) {
        val batch = Batch(neuralNetwork)
        trainingData.forEach { (input, target) -> batch.backPropagate(input, target) }
        batch.deltaBiases.forEach { it *= learningRate / trainingData.size }
        batch.deltaWeights.forEach { it *= learningRate / trainingData.size }
        neuralNetwork.layers.forEachIndexed { i, layer -> layer.bias -= batch.deltaBiases[i] }
        neuralNetwork.weights.forEachIndexed { i, weights -> weights -= batch.deltaWeights[i] }
    }

    private fun Batch.backPropagate(input: Vector, target: Vector) {
        if (input.size != neuralNetwork.layers.first().size) throw IllegalArgumentException("Invalid input size")
        if (target.size != neuralNetwork.layers.last().size) throw IllegalArgumentException("Invalid target size")

        val feedForwardResult = computer.feedForward(input)
        val weightedSums = feedForwardResult.weightedSums
        val activations = feedForwardResult.activations

        var delta = hadamard(
            costFunction.derivative(target, feedForwardResult.output),
            neuralNetwork.layers.last().activationFunction.derivative(weightedSums.last()),
        )
        deltaBiases[numLayers - 1].plusAssign(delta)
        deltaWeights[numLayers - 1] += outer(delta, activations[numLayers - 2])

        for (l in 2 until numLayers) {
            val activationFunction = neuralNetwork.layers[numLayers - l].activationFunction
            val sigmoidPrimes = activationFunction.derivative(weightedSums[numLayers - l])
            delta = hadamard(transposeDot(neuralNetwork.weights[numLayers - l + 1], delta), sigmoidPrimes)
            deltaBiases[numLayers - l].plusAssign(delta)
            deltaWeights[numLayers - l] += outer(delta, activations[numLayers - l - 1])
        }
    }
}
