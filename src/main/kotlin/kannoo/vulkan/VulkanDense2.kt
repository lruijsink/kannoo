package kannoo.vulkan

import kannoo.core.ActivationFunction

const val learningRate = 0.1f

interface VulkanLayer2 {
    val activation: ActivationFunction
    val input: VulkanMatrixBuffer
    val output: VulkanMatrixBuffer
    val deltaInput: VulkanMatrixBuffer?
    val forward: List<VulkanCommandBuffer>
    val back: List<VulkanCommandBuffer>
}

class VulkanDense2 : VulkanLayer2 {

    val vulkan: Vulkan = TODO()

    override val activation: ActivationFunction = TODO()

    override val input: VulkanMatrixBuffer = TODO()
    override val output: VulkanMatrixBuffer = TODO()
    override val deltaInput: VulkanMatrixBuffer? = TODO()

    val weights: VulkanMatrixBuffer = TODO()
    val bias: VulkanVectorBuffer = TODO()

    override val forward = listOf(
        vulkan.matMul(input, weights, output, transposeB = true),
        vulkan.matVecAdd(output, bias),
        vulkan.activate(output, output, activation),
    )

    override val back: List<VulkanCommandBuffer> = TODO()
}
