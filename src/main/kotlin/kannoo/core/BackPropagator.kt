package kannoo.core

import kannoo.impl.CrossEntropyLoss
import kannoo.impl.Softmax
import kannoo.math.hadamard

class BackPropagator(
    private val model: Model,
    private val cost: CostFunction,
) {
    init {
        if (model.layers.dropLast(1).any { it.activationFunction is Softmax })
            throw IllegalStateException("Softmax is only supported in the output layer")

        if ((model.layers.last().activationFunction is Softmax) xor (cost is CrossEntropyLoss))
            throw IllegalArgumentException("Cross-entropy loss requires Softmax and vice-versa")
    }

    fun backPropagate(sample: Sample): List<ParameterDeltas> {
        val parameterDeltas = mutableListOf<ParameterDeltas>()
        val forward = forwardPass(sample)
        var deltaOutput = cost.derivative(target = sample.target, actual = forward.last().output)

        for (i in model.layers.size - 1 downTo 0) {
            val deltaPreActivation =
                if (model.layers[i].activationFunction == Softmax) deltaOutput // Combined into one operation
                else hadamard(deltaOutput, model.layers[i].activationFunction.derivative(forward[i].preActivation))

            val backPropagation = model.layers[i].backPropagate(forward[i], deltaPreActivation)
            parameterDeltas += backPropagation.parameterDeltas
            deltaOutput = backPropagation.deltaInput
        }
        return parameterDeltas
    }

    private fun forwardPass(example: Sample): List<ForwardPass> {
        var input = example.input
        val results = mutableListOf<ForwardPass>()
        for (layer in model.layers) {
            val result = layer.forwardPass(input)
            results += result
            input = result.output
        }
        return results
    }
}
