package kannoo.impl

import kannoo.core.CostFunction
import kannoo.core.GradientComputer
import kannoo.core.Model
import kannoo.core.Sample
import kannoo.math.euclideanNorm

class MiniBatchSGD(
    private val model: Model,
    cost: CostFunction,
    private val learningRate: Float,
    private val batchSize: Int,
    private val maxNorm: Float = 1.0f,
) {
    private val gradientComputer = GradientComputer(model, cost)

    fun apply(samples: List<Sample<*, *>>) {
        samples.forEach {
            if (it.input.shape != model.inputLayer.shape)
                throw IllegalArgumentException("Sample has incorrect input shape: ${it.input.shape}, expected ${model.inputLayer.shape}")

            if (it.target.shape != model.layers.last().outputShape)
                throw IllegalArgumentException("Sample has incorrect target shape: ${it.target.shape}, expected ${model.layers.last().outputShape}")
        }
        samples.shuffled().chunked(batchSize).forEach(this::batch)
    }

    private fun batch(samples: List<Sample<*, *>>) {
        val gradients = gradientComputer.computeGradients(samples)
        val norm = euclideanNorm(gradients.values)

        val scale = learningRate / samples.size
        if (norm < maxNorm) learningRate / samples.size
        else learningRate / (samples.size * norm)

        for ((param, gradient) in gradients)
            param -= gradient * scale
    }
}
