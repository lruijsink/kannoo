package kannoo.core

class BackPropagator(
    private val model: Model,
    private val cost: CostFunction,
) {
    fun backPropagate(example: Sample): List<ParameterDeltas> {
        val parameterDeltas = mutableListOf<ParameterDeltas>()
        val forwardPasses = forwardPass(example)
        var deltaOutput = cost.derivative(example.target, forwardPasses.last().output)
        for (i in model.layers.size - 1 downTo 0) {
            val layer = model.layers[i]
            val forwardPass = forwardPasses[i]
            val backPropagation = layer.backPropagate(forwardPass, deltaOutput)
            parameterDeltas += backPropagation.parameterDeltas
            deltaOutput = backPropagation.deltaInput
        }
        return parameterDeltas
    }

    private fun forwardPass(example: Sample): List<ForwardPass> {
        var input = example.input
        val results = mutableListOf<ForwardPass>()
        for (layer in model.layers) {
            val result = layer.forwardPass(input)
            results += result
            input = result.output
        }
        return results
    }
}
