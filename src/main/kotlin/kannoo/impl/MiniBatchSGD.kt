package kannoo.impl

import kannoo.core.CostFunction
import kannoo.core.GradientComputer
import kannoo.core.Model
import kannoo.core.Sample
import kannoo.math.Composite
import kannoo.math.euclideanNorm

const val FORCE_BATCHING = true

class MiniBatchSGD(
    private val model: Model,
    cost: CostFunction,
    private val learningRate: Float,
    private val batchSize: Int,
    private val maxNorm: Float = 1.0f,
) {
    private val gradientComputer = GradientComputer(model, cost)

    fun train(samples: List<Sample>) {
        if (FORCE_BATCHING) return trainBatch(samples)

        samples.forEach {
            if (it.input.shape != model.inputLayer.shape)
                throw IllegalArgumentException("Sample has incorrect input shape: ${it.input.shape}, expected ${model.inputLayer.shape}")

            if (it.target.shape != model.layers.last().outputShape)
                throw IllegalArgumentException("Sample has incorrect target shape: ${it.target.shape}, expected ${model.layers.last().outputShape}")
        }
        samples.shuffled().chunked(batchSize).forEach(this::batch)
    }

    fun trainBatch(samples: List<Sample>) {
        samples.forEach {
            if (it.input.shape != model.inputLayer.shape)
                throw IllegalArgumentException("Sample has incorrect input shape: ${it.input.shape}, expected ${model.inputLayer.shape}")

            if (it.target.shape != model.layers.last().outputShape)
                throw IllegalArgumentException("Sample has incorrect target shape: ${it.target.shape}, expected ${model.layers.last().outputShape}")
        }
        samples.shuffled().chunked(batchSize).forEach(this::batched)
    }

    private fun batch(samples: List<Sample>) {
        val gradients = gradientComputer.computeGradients(samples)
        val norm = euclideanNorm(gradients.values)

        val scale =
            if (norm < maxNorm) learningRate / samples.size
            else learningRate / (samples.size * norm)

        for ((param, gradient) in gradients)
            param -= gradient * scale
    }

    private fun batched(samples: List<Sample>) {
        val inputs = Composite(samples.map { it.input })
        val targets = Composite(samples.map { it.target })

        val gradients = gradientComputer.computeGradientsBatch(inputs, targets)
        val norm = euclideanNorm(gradients.values)

        val scale =
            if (norm < maxNorm) learningRate / samples.size
            else learningRate / (samples.size * norm)

        for ((param, gradient) in gradients)
            param -= gradient * scale
    }
}
