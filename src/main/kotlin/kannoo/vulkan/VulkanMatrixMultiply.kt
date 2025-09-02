package kannoo.vulkan

import kannoo.math.Matrix
import kannoo.math.Shape
import org.lwjgl.system.MemoryStack.stackPush

class VulkanMatrixMultiply(
    private val vulkan: Vulkan,
    private val shader: Shader,
    private val inputSize: Int,
    private val outputSize: Int,
    private val batchSize: Int,
) {
    private val inputBufferSize: Long = batchSize * inputSize * Float.SIZE_BYTES.toLong()
    private val weightsBufferSize: Long = inputSize * outputSize * Float.SIZE_BYTES.toLong()
    private val outputBufferSize: Long = batchSize * outputSize * Float.SIZE_BYTES.toLong()

    private val inputBuffer: VulkanBuffer = vulkan.createBuffer(inputBufferSize)
    private val weightsBuffer: VulkanBuffer = vulkan.createBuffer(weightsBufferSize)
    private val outputBuffer: VulkanBuffer = vulkan.createBuffer(outputBufferSize)

    private val descriptorSet = vulkan.createDescriptorSet(
        0 to inputBuffer,
        1 to weightsBuffer,
        2 to outputBuffer,
    )

    private val pushConstants = VulkanPushConstants(
        inputSize,
        outputSize,
        batchSize,
    )

    private val pipeline = vulkan.createPipeline(
        descriptorSet,
        pushConstants,
        shader,
    )

    private val commandBuffer = vulkan.createCommandBuffer(
        pipeline,
        (outputSize + shader.workgroupSize - 1) / shader.workgroupSize,
        (batchSize + shader.workgroupSize - 1) / shader.workgroupSize,
        1,
    )

    private val execution = vulkan.createExecution(
        commandBuffer,
    )

    fun multiplyTransposed(left: Matrix, right: Matrix, destination: Matrix) {
        if (left.shape != Shape(batchSize, inputSize))
            throw IllegalArgumentException("Expecting $batchSize x $inputSize left argument, got ${left.shape}")

        if (right.shape != Shape(outputSize, inputSize))
            throw IllegalArgumentException("Expecting $outputSize x $inputSize right argument, got ${right.shape}")

        if (destination.shape != Shape(batchSize, outputSize))
            throw IllegalArgumentException("Expecting $batchSize x $outputSize destination, got ${destination.shape}")

        setInput(left)
        setWeights(right)
        runCommandBuffer()
        getOutput(destination)
    }

    fun multiplyTransposed(left: Matrix, right: Matrix): Matrix {
        val destination = Matrix(batchSize, outputSize)
        multiplyTransposed(left, right, destination)
        return destination
    }

    private fun setWeights(weights: Matrix): Unit = stackPush().use { stack ->
        val buffer = weightsBuffer.mapped.asFloatBuffer()
        weights.slices.forEach { buffer.put(it.elements) }
        buffer.flip()
    }

    private fun setInput(batch: Matrix): Unit = stackPush().use { stack ->
        val buffer = inputBuffer.mapped.asFloatBuffer()
        batch.slices.forEach { buffer.put(it.elements) }
        buffer.flip()
    }

    private fun runCommandBuffer() {
        execution.submit()
    }

    private fun getOutput(destination: Matrix): Unit = stackPush().use { stack ->
        val buffer = outputBuffer.mapped.asFloatBuffer()
        for (i in 0 until batchSize) buffer.get(destination.slices[i].elements)
        buffer.flip()
    }

    fun destroy() {
        inputBuffer.destroy()
        weightsBuffer.destroy()
        outputBuffer.destroy()
        descriptorSet.destroy()
        pipeline.destroy()
        commandBuffer.destroy()
        execution.destroy()
    }
}
