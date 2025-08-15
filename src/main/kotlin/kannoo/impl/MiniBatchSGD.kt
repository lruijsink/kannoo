package kannoo.impl

import kannoo.core.CostFunction
import kannoo.core.GradientComputer
import kannoo.core.Model
import kannoo.core.Sample

class MiniBatchSGD(
    model: Model,
    cost: CostFunction,
    private val learningRate: Float,
    private val batchSize: Int,
) {
    private val gradientComputer = GradientComputer(model, cost)

    fun apply(samples: List<Sample>) {
        samples.shuffled().chunked(batchSize).forEach(this::batch)
    }

    private fun batch(samples: List<Sample>) {
        for ((param, gradient) in gradientComputer.computeGradients(samples))
            param -= gradient * (learningRate / samples.size)
    }
}
