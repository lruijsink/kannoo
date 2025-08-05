package kannoo.old

import kannoo.math.Matrix
import kannoo.math.emptyMatrix
import kannoo.math.randomMatrix

class NeuralNetwork(
    val layers: List<Layer>,
) {
    init {
        if (layers.size < 2) throw IllegalArgumentException("Too few layers")
    }

    val weights: List<Matrix> = List(layers.size) { i ->
        if (i == 0) emptyMatrix()
        else randomMatrix(layers[i].size, layers[i - 1].size)
    }
}
