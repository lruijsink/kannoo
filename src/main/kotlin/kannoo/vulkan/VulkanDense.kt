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
    val deltaInput: VulkanMatrixBuffer,
    val deltaOutput: VulkanMatrixBuffer,
    val activation: ActivationFunction,
) : VulkanLayer, VulkanResource(vulkan) {

    val inputSize: Int = input.cols
    val outputSize: Int = output.cols
    val weights: VulkanMatrixBuffer = vulkan.createMatrixBuffer(outputSize, inputSize)
    val bias: VulkanVectorBuffer = vulkan.createVectorBuffer(outputSize)

    override val inputBuffer: VulkanBuffer = input.buffer
    override val outputBuffer: VulkanBuffer = output.buffer

    override val forward: List<VulkanCommandBuffer> = listOf(
        vulkan.matMul(input, weights, output, transposeB = true),
        vulkan.matVecAdd(output, bias),
        vulkan.activate(output.buffer, activation),
    )

    override val backProp: List<VulkanCommandBuffer> = listOf(
        vulkan.activate(deltaOutput.buffer, activation, derivative = true),
        vulkan.matMul(deltaOutput, weights, deltaInput),
        vulkan.matMulAcc(deltaOutput, input, weights, transposeA = true),
        vulkan.matRowAcc(deltaOutput, bias),
    )

    override fun recordBackProp(learningRate: Float) {
        backProp[2].record(pushConstants(learningRate))
        backProp[3].record(pushConstants(learningRate))
    }
}
