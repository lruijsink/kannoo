package kannoo.impl

import kannoo.core.BackPropagator
import kannoo.core.CostFunction
import kannoo.core.GradientReceiver
import kannoo.core.Model
import kannoo.core.Sample

class BatchSGD(
    model: Model,
    cost: CostFunction,
    private val learningRate: Float,
    private val batchSize: Int,
) {
    private val backPropagator = BackPropagator(model, cost)
    private val gradientReceiver = GradientReceiver(model)

    fun apply(samples: List<Sample>) {
        samples.shuffled().chunked(batchSize).forEach(this::batch)
    }

    private fun batch(samples: List<Sample>) {
        gradientReceiver.reset()
        samples.forEach {
            backPropagator.calculatePartials(it, gradientReceiver)
        }
        gradientReceiver.apply { param, gradient ->
            param -= gradient * (learningRate / samples.size)
        }
    }
}
