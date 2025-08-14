package kannoo.impl

import kannoo.core.BackPropagator
import kannoo.core.CostFunction
import kannoo.core.Model
import kannoo.core.Sample
import kannoo.math.Matrix
import kannoo.math.Vector

class BatchSGD(
    private val model: Model,
    cost: CostFunction,
    private val learningRate: Float,
    private val batchSize: Int,
) {
    private val backpropagator = BackPropagator(model, cost)

    private val deltaWeights = model.layers.map {
        Matrix((it as DenseLayer /* TODO: generalize */).weights.rows, it.weights.cols)
    }

    private val deltaBias = model.layers.map {
        Vector(it.size)
    }

    fun apply(samples: List<Sample>) {
        samples.shuffled().chunked(batchSize).forEach(this::batch)
    }

    private fun batch(samples: List<Sample>) {
        deltaWeights.forEach { it.zero() }
        deltaBias.forEach { it.zero() }
        samples.forEach { sample ->
            backpropagator.calculatePartials(sample) { i, dW, db ->
                deltaWeights[i] += dW
                deltaBias[i] += db
            }
        }
        for (i in 0 until model.layers.size) {
            val l = model.layers[i] as DenseLayer // TODO: generalize
            l.weights.minusAssign(deltaWeights[i] * (learningRate / batchSize))
            l.bias.minusAssign(deltaBias[i] * (learningRate / batchSize))
        }
    }
}
