package kannoo.vulkan

import kannoo.core.ActivationFunction
import kannoo.impl.MeanSquaredError
import kannoo.math.Matrix

class DenseConfig(
    val size: Int,
    val activation: ActivationFunction,
)

class VulkanModel(
    vulkan: Vulkan,
    inputSize: Int,
    batchSize: Int,
    configs: List<DenseConfig>,
) {
    val denseLayers: List<VulkanDense>

    init {
        val inputs = mutableListOf<VulkanMatrixBuffer>()
        val outputs = mutableListOf<VulkanMatrixBuffer>()

        configs.forEachIndexed { i, config ->
            inputs += if (i == 0) vulkan.createMatrixBuffer(batchSize, inputSize) else outputs[i - 1]
            outputs += vulkan.createMatrixBuffer(batchSize, config.size)
        }

        val deltaOutputs = outputs.map {
            vulkan.createMatrixBuffer(it.rows, it.cols)
        }

        val deltaInputs = inputs.mapIndexed { i, x ->
            if (i == 0) null
            else deltaOutputs[i - 1]
        }

        denseLayers = configs.mapIndexed { i, config ->
            VulkanDense(
                vulkan,
                inputs[i],
                outputs[i],
                deltaInputs[i],
                deltaOutputs[i],
                config.activation,
            )
        }
    }

    val forward = vulkan.createExecution(denseLayers.flatMap { it.forward })
    val backProp = vulkan.createExecution(denseLayers.reversed().flatMap { it.backProp })

    fun print() {
        denseLayers.forEachIndexed { i, l ->
            println("-----------------------------------")
            println("Layer $i")
            println()

            println("Bias: ")
            println(l.bias.get().prettyPrint())
            println()

            println("Input: ")
            println(l.input.get().prettyPrint())
            println()

            println("Weights: ")
            println(l.weights.get().prettyPrint())
            println()

            println("Pre-activation: ")
            println(l.preActivation.get().prettyPrint())
            println()

            println("Output: ")
            println(l.output.get().prettyPrint())
            println()
        }
    }

    fun compute(input: Matrix): Matrix {
        denseLayers.first().input.set(input)
        forward.submit()
        return denseLayers.last().output.get()
    }

    fun backProp(input: Matrix, target: Matrix, learningRate: Float) {
        denseLayers.first().input.set(input)
        forward.submit()
        val actual = denseLayers.last().output.get()

        val deltaOutput = MeanSquaredError.derivative(target, actual)
        denseLayers.forEach { it.recordBackProp(learningRate) }
        denseLayers.last().deltaOutput.set(deltaOutput)
        backProp.submit()
    }
}
