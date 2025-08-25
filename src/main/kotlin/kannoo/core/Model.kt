package kannoo.core

import kannoo.math.Tensor

class Model {
    val inputLayer: InputLayer
    val layers: List<InnerLayer>

    constructor(inputLayer: InputLayer, layers: List<InnerLayer>) {
        this.inputLayer = inputLayer
        this.layers = layers
    }

    constructor(inputLayer: InputLayer, vararg initializers: InnerLayerInitializer) {
        this.inputLayer = inputLayer

        val innerLayerAcc = mutableListOf<InnerLayer>()
        initializers.forEachIndexed { i, init ->
            innerLayerAcc += init.initialize(
                if (i == 0) inputLayer.shape
                else innerLayerAcc[i - 1].outputShape,
            )
        }

        this.layers = innerLayerAcc
    }

    constructor(inputLayer: InputLayer, vararg layers: InnerLayer) :
            this(inputLayer, layers.toList())

    fun compute(input: Tensor): Tensor =
        layers.fold(input) { v, layer -> layer.compute(v) }
}
