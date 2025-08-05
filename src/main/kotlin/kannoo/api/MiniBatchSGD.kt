package kannoo.api

import kannoo.CostFunction
import kannoo.sumOfMatrix
import kannoo.sumOfVector

class MiniBatchSGD(
    model: Model,
    cost: CostFunction,
    private val batchSize: Int,
    private val learningRate: Double,
) {
    private val backPropagator = BackPropagator(model, cost)

    fun apply(trainingData: List<TrainingExample>) {
        trainingData.shuffled().chunked(batchSize).forEach(::miniBatch)
    }

    private fun miniBatch(batch: List<TrainingExample>) {
        val parameterDeltas = batch.flatMap { backPropagator.backPropagate(it) }
        val matrices = parameterDeltas.flatMap { it.matrices }
        val vectors = parameterDeltas.flatMap { it.vectors }
        val scale = learningRate / batch.size

        matrices.groupBy { it.param }.forEach { (param, paramMatrices) ->
            param -= paramMatrices.sumOfMatrix { it.delta } * scale
        }

        vectors.groupBy { it.param }.forEach { (param, paramVectors) ->
            param -= paramVectors.sumOfVector { it.delta } * scale
        }
    }
}
