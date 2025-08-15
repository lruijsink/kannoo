package kannoo.core

import kannoo.math.Vector

class Model(
    val inputLayer: InputLayer,
    val layers: List<InnerLayer>,
) {
    constructor(inputLayer: InputLayer, vararg layers: InnerLayer) : this(inputLayer, layers.toList())

    init {
        layers.forEachIndexed { i, layer ->
            layer.initialize(if (i == 0) inputLayer.size else layers[i - 1].size)
        }
    }

    fun compute(input: Vector): Vector =
        layers.fold(input) { v, layer -> layer.compute(v) }
}
