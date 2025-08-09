package kannoo.impl

import kannoo.core.BackPropagator
import kannoo.core.CostFunction
import kannoo.core.Model
import kannoo.core.Sample
import kannoo.math.sumOfTensor

class SGD(
    model: Model,
    cost: CostFunction,
    private val learningRate: Float,
) {
    private val backPropagator = BackPropagator(model, cost)

    fun apply(trainingData: List<Sample>) {
        trainingData.shuffled().forEach { apply(it) }
    }

    private fun apply(sample: Sample) {
        val parameterDeltas = backPropagator.backPropagate(sample)
        val matrices = parameterDeltas.flatMap { it.matrices }
        val vectors = parameterDeltas.flatMap { it.vectors }

        matrices.groupBy { it.param }.forEach { (param, paramMatrices) ->
            param -= paramMatrices.sumOfTensor { it.delta } * learningRate
        }

        vectors.groupBy { it.param }.forEach { (param, paramVectors) ->
            param -= paramVectors.sumOfTensor { it.delta } * learningRate
        }
    }
}
