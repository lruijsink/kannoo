package kannoo.api

import kannoo.ActivationFunction
import kannoo.Matrix
import kannoo.Vector

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
        layers.fold(input) { v, layer -> layer.forwardPass(v).output }
}

class InputLayer(
    val size: Int,
)

class ForwardPass(
    val input: Vector,
    val preActivation: Vector,
    val output: Vector,
)

abstract class InnerLayer(
    val size: Int,
    val activationFunction: ActivationFunction,
) {
    abstract fun initialize(previousLayerSize: Int)

    abstract fun forwardPass(input: Vector): ForwardPass

    abstract fun backPropagate(forwardPass: ForwardPass, deltaOutput: Vector): BackPropagation
}

class BackPropagation(
    val deltaInput: Vector,
    val parameterDeltas: ParameterDeltas,
)

class ParameterDelta<T : Any>(val param: T, val delta: T)

class ParameterDeltas(
    val matrices: List<ParameterDelta<Matrix>> = listOf(),
    val vectors: List<ParameterDelta<Vector>> = listOf(),
)
