package kannoo.vulkan

import kannoo.core.ActivationFunction
import kannoo.math.randomMatrix

class VulkanMatrixMul2(
    val vulkan: Vulkan,
    val input: VulkanMatrixBuffer,
    val weights: VulkanMatrixBuffer,
    val output: VulkanMatrixBuffer,
    val activation: ActivationFunction,
) {
    constructor(
        vulkan: Vulkan,
        input: VulkanMatrixBuffer,
        outputSize: Int,
        activation: ActivationFunction,
    ) : this(
        vulkan = vulkan,
        input = input,
        weights = vulkan.createMatrixBuffer(rows = outputSize, cols = input.cols),
        output = vulkan.createMatrixBuffer(rows = input.rows, cols = outputSize),
        activation = activation,
    )

    val shader = vulkan.createShader("shaders/tiled.spv", 32)

    val inputSize = input.cols
    val outputSize = output.cols
    val batchSize = output.rows

    val descriptorSet = vulkan.createDescriptorSet(
        0 to input.buffer,
        1 to weights.buffer,
        2 to output.buffer,
    )

    val pushConstants = VulkanPushConstants(
        inputSize,
        outputSize,
        batchSize,
    )

    val pipeline = vulkan.createPipeline(
        descriptorSet,
        pushConstants,
        shader,
        createSpecialization(
            0 to inputSize,
            1 to outputSize,
            2 to batchSize,
        ),
    )

    val commandBuffer = vulkan.createCommandBuffer(
        pipeline = pipeline,
        groupCountX = (outputSize + shader.workgroupSize - 1) / shader.workgroupSize,
        groupCountY = (batchSize + shader.workgroupSize - 1) / shader.workgroupSize,
    )

    val execution = vulkan.createExecution(
        commandBuffer,
    )

    init {
        if (input.rows != output.rows)
            throw IllegalArgumentException("Input and output must have same number of rows (= batch size)")

        weights.set(randomMatrix(weights.rows, weights.cols))
    }

    fun runCommandBuffer() {
        execution.submit()
    }
}
