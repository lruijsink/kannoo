package kannoo.vulkan

import kannoo.core.ActivationFunction
import kannoo.vulkan.ActivationMode.INFER_DERIVATIVE

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
//    val preActivation: VulkanMatrixBuffer = vulkan.createMatrixBuffer(output.rows, output.cols)

    override val inputBuffer: VulkanBuffer = input.buffer
    override val outputBuffer: VulkanBuffer = output.buffer

    override val forward: List<VulkanCommandBuffer> = listOf(
        vulkan.matMul(input, weights, output, transposeB = true),
        vulkan.matVecAdd(output, bias),
        vulkan.activateAssign(output, activation),
    )

    override val backProp: List<VulkanCommandBuffer> =
        if (deltaInput != null)
            listOf(
                vulkan.activateAssign(output, activation, INFER_DERIVATIVE),
                vulkan.hadamardAssign(output, deltaOutput),
                vulkan.matMul(output, weights, deltaInput),
                vulkan.matMulAcc(output, input, weights, transposeA = true),
                vulkan.matRowAcc(output, bias),
            )
        else
            listOf(
                vulkan.activateAssign(output, activation, INFER_DERIVATIVE),
                vulkan.hadamardAssign(output, deltaOutput),
                vulkan.matMulAcc(output, input, weights, transposeA = true),
                vulkan.matRowAcc(output, bias),
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
