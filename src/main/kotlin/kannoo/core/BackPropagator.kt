package kannoo.core

import kannoo.impl.Softmax
import kannoo.math.Tensor

class BackPropagator(
    val model: Model,
    val cost: CostFunction,
) {
    private val preActivations = model.layers.map { it.outputShape.createTensor() }.toMutableList()
    private val activations = model.layers.map { it.outputShape.createTensor() }.toMutableList()

    fun calculatePartials(sample: Sample<*, *>, gradientReceiver: GradientReceiver) {
        forwardPass(sample)
        backPropagate(sample, gradientReceiver)
    }

    private fun forwardPass(sample: Sample<*, *>) {
        var input = sample.input as Tensor
        model.layers.forEachIndexed { i, layer ->
            preActivations[i] = layer.preActivation(input)
            activations[i] = layer.activationFunction.compute(preActivations[i])
            input = activations[i]
        }
    }

    private fun backPropagate(sample: Sample<*, *>, gradientReceiver: GradientReceiver) {
        var deltaActivation = cost.derivative(sample.target, activations[model.layers.size - 1])

        for (i in (model.layers.size - 1) downTo 0) {
            val deltaPreActivation =
                if (model.layers[i].activationFunction == Softmax) deltaActivation // Combined into one operation
                else deltaActivation.hadamard(model.layers[i].activationFunction.derivative(preActivations[i]))

            val input =
                if (i == 0) sample.input
                else activations[i - 1]

            if (i > 0)
                deltaActivation = model.layers[i].deltaInput(deltaPreActivation, input)

            model.layers[i].gradients(deltaPreActivation, input, gradientReceiver)
        }
    }
}
