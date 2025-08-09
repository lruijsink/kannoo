package kannoo.impl

import kannoo.core.BackPropagator
import kannoo.core.CostFunction
import kannoo.core.Model
import kannoo.core.Sample
import kannoo.math.sumOfTensor

class MiniBatchSGD(
    model: Model,
    cost: CostFunction,
    private val batchSize: Int,
    private val learningRate: Float,
) {
    private val backPropagator = BackPropagator(model, cost)

    fun apply(trainingData: List<Sample>) {
        trainingData.shuffled().chunked(batchSize).forEach(::miniBatch)
    }

    private fun miniBatch(batch: List<Sample>) {
        val parameterDeltas = batch.flatMap { backPropagator.backPropagate(it) }
        val matrices = parameterDeltas.flatMap { it.matrices }
        val vectors = parameterDeltas.flatMap { it.vectors }
        val scale = learningRate / batch.size

        matrices.groupBy { it.param }.forEach { (param, paramMatrices) ->
            param -= paramMatrices.sumOfTensor { it.delta } * scale
        }

        vectors.groupBy { it.param }.forEach { (param, paramVectors) ->
            param -= paramVectors.sumOfTensor { it.delta } * scale
        }
    }
}
