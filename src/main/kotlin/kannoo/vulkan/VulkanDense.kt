package kannoo.vulkan

import kannoo.core.ActivationFunction

interface VulkanLayer {
    val inputBuffer: VulkanBuffer
    val outputBuffer: VulkanBuffer

    val forward: List<VulkanCommandBuffer>
    val backProp: List<VulkanCommandBuffer>

    fun recordBackProp(learningRate: Float)
}

class VulkanDense(
    vulkan: Vulkan,
    val input: VulkanMatrixBuffer,
    val output: VulkanMatrixBuffer,
    val deltaInput: VulkanMatrixBuffer? = null,
    val deltaOutput: VulkanMatrixBuffer,
    val activation: ActivationFunction,
) : VulkanLayer, VulkanResource(vulkan) {

    val inputSize: Int = input.cols
    val outputSize: Int = output.cols
    val weights: VulkanMatrixBuffer = vulkan.createMatrixBuffer(outputSize, inputSize).randomize()
    val bias: VulkanVectorBuffer = vulkan.createVectorBuffer(outputSize).zero()
    val preActivation: VulkanMatrixBuffer = vulkan.createMatrixBuffer(output.rows, output.cols)

    override val inputBuffer: VulkanBuffer = input.buffer
    override val outputBuffer: VulkanBuffer = output.buffer

    override val forward: List<VulkanCommandBuffer> = listOf(
        vulkan.matMul(input, weights, preActivation, transposeB = true),
        vulkan.matVecAdd(preActivation, bias),
        vulkan.activate(preActivation, output, activation),
    )

    override val backProp: List<VulkanCommandBuffer> =
        if (deltaInput != null)
            listOf(
                vulkan.activateAssign(preActivation, activation, derivative = true),
                vulkan.hadamardAssign(preActivation, deltaOutput),
                vulkan.matMul(preActivation, weights, deltaInput),
                vulkan.matMulAcc(preActivation, input, weights, transposeA = true),
                vulkan.matRowAcc(preActivation, bias),
            )
        else
            listOf(
                vulkan.activateAssign(preActivation, activation, derivative = true),
                vulkan.hadamardAssign(preActivation, deltaOutput),
                vulkan.matMulAcc(preActivation, input, weights, transposeA = true),
                vulkan.matRowAcc(preActivation, bias),
            )

    override fun recordBackProp(learningRate: Float) {
        if (deltaInput != null) {
            backProp[3].record(pushConstants(learningRate))
            backProp[4].record(pushConstants(learningRate))
        } else {
            backProp[2].record(pushConstants(learningRate))
            backProp[3].record(pushConstants(learningRate))
        }
    }
}
