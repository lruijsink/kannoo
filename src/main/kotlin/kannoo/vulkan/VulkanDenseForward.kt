package kannoo.vulkan

import org.lwjgl.system.MemoryUtil.memFree

class VulkanDenseForward(
    val vulkan: Vulkan,
    val input: VulkanMatrixBuffer,
    val output: VulkanMatrixBuffer,
) {
    init {
        if (input.rows != output.rows)
            throw IllegalArgumentException("Input and output must have same number of rows (= batch size)")
    }

    val weights = vulkan.createMatrixBuffer(rows = output.cols, cols = input.cols)

    val bias = vulkan.createVectorBuffer(size = output.cols)

    val descriptorSet = vulkan.createDescriptorSet(
        0 to input.buffer,
        1 to weights.buffer,
        2 to bias.buffer,
        3 to output.buffer,
    )

    val pushConstants = VulkanPushConstants(
        input.cols,
        output.cols,
        input.rows,
    )

    val shader = Shader(fileName = "shaders/dense_forward.spv", workgroupSize = 32)

    val pipeline = vulkan.createPipeline(
        descriptorSet,
        pushConstants,
        shader,
    )

    val commandBuffer = vulkan.createCommandBuffer(
        pipeline,
        groupCountX = (output.cols + shader.workgroupSize - 1) / shader.workgroupSize,
        groupCountY = (output.rows + shader.workgroupSize - 1) / shader.workgroupSize,
    )

    val execution = vulkan.createExecution(
        commandBuffer,
    )

    fun forward() {
        // TODO
    }

    fun destroy() {
        execution.destroy()
        commandBuffer.destroy()
        pipeline.destroy()
        memFree(shader.code)
        descriptorSet.destroy()
    }
}
