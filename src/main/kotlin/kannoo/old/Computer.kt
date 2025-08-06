package kannoo.old

import kannoo.math.Vector

class Computer(val neuralNetwork: NeuralNetwork) {
    private val weightedSums = neuralNetwork.layers.map { Vector(it.size) }
    private val activations = neuralNetwork.layers.map { Vector(it.size) }

    fun compute(input: Vector): Vector =
        feedForward(input).output

    fun feedForward(input: Vector): FeedForwardResult {
        if (input.size != activations[0].size) throw IllegalArgumentException("Invalid input size")
        input.copyInto(activations[0])
        for (p in 1 until activations.size) {
            for (i in 0 until activations[p].size) {
                weightedSums[p][i] = neuralNetwork.layers[p].bias[i]
                for (j in 0 until activations[p - 1].size)
                    weightedSums[p][i] += activations[p - 1][j] * neuralNetwork.weights[p][i][j]
                neuralNetwork.layers[p].activationFunction.compute(weightedSums[p]).copyInto(activations[p])
            }
        }
        return FeedForwardResult(weightedSums, activations)
    }
}

class FeedForwardResult(
    val weightedSums: List<Vector>,
    val activations: List<Vector>,
) {
    val output = activations.last()
}
