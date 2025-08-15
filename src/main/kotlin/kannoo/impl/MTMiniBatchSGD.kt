package kannoo.impl

import kannoo.core.BackPropagator
import kannoo.core.CostFunction
import kannoo.core.Model
import kannoo.core.Sample
import kannoo.math.Matrix
import kannoo.math.Vector

/**
 * Multi-threaded mini-batch SGD
 */
class MTMiniBatchSGD(
    private val model: Model,
    cost: CostFunction,
    private val learningRate: Float,
    private val batchSize: Int,
    private val threadCount: Int = 8,
    private val maxNorm: Float = 1f,
) {
    private val backpropagator = BackPropagator(model, cost)

    private val deltaWeights = List(threadCount) {
        model.layers.map {
            Matrix((it as DenseLayer /* TODO: generalize */).weights.rows, it.weights.cols)
        }
    }

    private val deltaBias = List(threadCount) {
        model.layers.map {
            Vector(it.size)
        }
    }

    fun apply(samples: List<Sample>) {
        samples.shuffled().chunked(batchSize).forEach(this::batch)
    }

    private fun batch(samples: List<Sample>): Unit = TODO() /*{
        val threads = samples.chunked(batchSize / threadCount).mapIndexed { t, chunk ->
            Thread {
                deltaWeights[t].forEach { it.zero() }
                deltaBias[t].forEach { it.zero() }
                chunk.forEach { sample ->
                    backpropagator.calculatePartials(sample) { i, dW, db ->
                        deltaWeights[t][i] += dW
                        deltaBias[t][i] += db
                    }
                }
            }
        }
        threads.forEach { it.start() }
        threads.forEach { it.join() }
        for (i in 0 until model.layers.size) {
            val l = model.layers[i] as DenseLayer // TODO: Generalize
            val dW = deltaWeights.sumOfTensor { it[i] } / batchSize.toFloat()
            val db = deltaBias.sumOfTensor { it[i] } / batchSize.toFloat()
            val norm = sqrt(square(dW).sum() + square(db).sum())
            if (norm > maxNorm) {
                dW /= norm
                db /= norm
            }
            l.weights.minusAssign(dW * learningRate)
            l.bias.minusAssign(db * learningRate)
        }
    }*/
}
