package kannoo.core

import kannoo.impl.Softmax
import kannoo.math.Composite
import kannoo.math.Shape

class BackPropagator(
    val model: Model,
    val cost: CostFunction,
) {
    private val preActivations = model.layers.map { it.outputShape.createTensor() }.toMutableList()
    private val activations = model.layers.map { it.outputShape.createTensor() }.toMutableList()

    fun calculatePartials(sample: Sample, gradientReceiver: GradientReceiver) {
        forwardPass(sample)
        backPropagate(sample, gradientReceiver)
    }

    private fun forwardPass(sample: Sample) {
        var input = sample.input
        model.layers.forEachIndexed { i, layer ->
            preActivations[i] = layer.preActivation(input)
            activations[i] = layer.activationFunction.compute(preActivations[i])
            input = activations[i]
        }
    }

    private fun backPropagate(sample: Sample, gradientReceiver: GradientReceiver) {
        var deltaActivation = cost.derivative(sample.target, activations.last())

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

    fun calculatePartialsBatch(inputs: Composite, targets: Composite, gradientReceiver: GradientReceiver) {
        val batchSize = inputs.size

        val batchPreActivations = model.layers
            .map { Shape(batchSize, it.outputShape).createTensor() as Composite }
            .toMutableList()

        val batchActivations = model.layers
            .map { Shape(batchSize, it.outputShape).createTensor() as Composite }
            .toMutableList()

        forwardPassBatch(inputs, batchPreActivations, batchActivations)
        backPropagateBatch(inputs, targets, gradientReceiver, batchPreActivations, batchActivations)
    }

    private fun forwardPassBatch(
        inputs: Composite,
        batchPreActivations: MutableList<Composite>,
        batchActivations: MutableList<Composite>,
    ) {
        var intermediate = inputs
        model.layers.forEachIndexed { i, layer ->
            batchPreActivations[i] = layer.preActivationBatch(intermediate)
            batchActivations[i] = layer.activationFunction.compute(batchPreActivations[i]) as Composite
            intermediate = batchActivations[i]
        }
    }

    private fun backPropagateBatch(
        inputs: Composite,
        targets: Composite,
        gradientReceiver: GradientReceiver,
        batchPreActivations: MutableList<Composite>,
        batchActivations: MutableList<Composite>,
    ) {
        var deltaActivations = cost.derivative(targets, batchActivations.last()) as Composite

        for (i in (model.layers.size - 1) downTo 0) {
            val deltaPreActivations =
                if (model.layers[i].activationFunction == Softmax) deltaActivations // Combined into one operation
                else deltaActivations.hadamard(model.layers[i].activationFunction.derivative(batchPreActivations[i])) as Composite

            val input =
                if (i == 0) inputs
                else batchActivations[i - 1]

            if (i > 0)
                deltaActivations = model.layers[i].deltaInputBatch(deltaPreActivations, input)

            model.layers[i].gradientsBatch(deltaPreActivations, input, gradientReceiver)
        }
    }
}
