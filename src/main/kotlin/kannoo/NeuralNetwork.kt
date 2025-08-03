package kannoo

class NeuralNetwork(
    layerSizes: List<Int>,
    val activationFunction: ActivationFunction,
    val costFunction: CostFunction,
) {
    class Layer(val size: Int) {
        val bias = randomVector(size)
    }

    init {
        if (layerSizes.size < 2) throw IllegalArgumentException("Too few layers")
    }

    val layers: List<Layer> = layerSizes.map { Layer(it) }
    val weights: List<Matrix> = List(layers.size) { i ->
        if (i == 0) emptyMatrix()
        else randomMatrix(layers[i].size, layers[i - 1].size)
    }
}
