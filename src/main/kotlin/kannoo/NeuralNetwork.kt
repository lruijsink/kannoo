package kannoo

class NeuralNetwork(
    val layers: List<Layer>,
    val costFunction: CostFunction,
) {
    init {
        if (layers.size < 2) throw IllegalArgumentException("Too few layers")
    }

    val weights: List<Matrix> = List(layers.size) { i ->
        if (i == 0) emptyMatrix()
        else randomMatrix(layers[i].size, layers[i - 1].size)
    }
}
