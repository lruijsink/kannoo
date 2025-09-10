package kannoo.vulkan

import kannoo.core.ActivationFunction

enum class ActivationMode(val shaderValue: Int) {
    REGULAR(0),
    DERIVATIVE(1),
    INFER_DERIVATIVE(2),
}

fun Vulkan.activate(
    input: VulkanBuffer,
    output: VulkanBuffer,
    activation: ActivationFunction,
    mode: ActivationMode = ActivationMode.REGULAR,
): VulkanCommandBuffer {
    if (input.size != output.size) throw IllegalArgumentException("Input and output buffers must have same size")
    val elements = (input.size / Float.SIZE_BYTES).toInt()
    return createCommandBuffer(
        pipeline = createPipeline(
            shader = createShader("shaders/el_activate.spv"),
            descriptorSet = createDescriptorSet(
                0 to input,
                1 to output,
            ),
            specialization = createSpecialization(
                0 to elements,
                1 to activation.vulkanShaderId,
                2 to mode.shaderValue,
            ),
        ),
        groupCountX = elements.divCeil(64),
    )
}

fun Vulkan.activate(
    input: VulkanMatrixBuffer,
    output: VulkanMatrixBuffer,
    activation: ActivationFunction,
    mode: ActivationMode = ActivationMode.REGULAR,
): VulkanCommandBuffer =
    activate(input.buffer, output.buffer, activation, mode)

fun Vulkan.activateAssign(
    buffer: VulkanBuffer,
    activation: ActivationFunction,
    mode: ActivationMode = ActivationMode.REGULAR,
): VulkanCommandBuffer =
    activate(buffer, buffer, activation, mode)

fun Vulkan.activateAssign(
    buffer: VulkanMatrixBuffer,
    activation: ActivationFunction,
    mode: ActivationMode = ActivationMode.REGULAR,
): VulkanCommandBuffer =
    activate(buffer, buffer, activation, mode)

fun Vulkan.hadamard(
    bufferA: VulkanBuffer,
    bufferB: VulkanBuffer,
    bufferOut: VulkanBuffer,
): VulkanCommandBuffer {
    if (bufferA.size != bufferB.size || bufferA.size != bufferOut.size)
        throw IllegalArgumentException("All buffers must have same size for elementwise multiplication")

    val elements = (bufferA.size / Float.SIZE_BYTES).toInt()
    return createCommandBuffer(
        pipeline = createPipeline(
            shader = createShader("shaders/el_mul.spv"),
            descriptorSet = createDescriptorSet(
                0 to bufferA,
                1 to bufferB,
                2 to bufferOut,
            ),
            specialization = createSpecialization(
                0 to elements,
            ),
        ),
        groupCountX = elements.divCeil(64),
    )
}

fun Vulkan.hadamard(
    bufferA: VulkanMatrixBuffer,
    bufferB: VulkanMatrixBuffer,
    bufferOut: VulkanMatrixBuffer,
): VulkanCommandBuffer =
    hadamard(bufferA.buffer, bufferB.buffer, bufferOut.buffer)

fun Vulkan.hadamardAssign(bufferA: VulkanBuffer, bufferB: VulkanBuffer): VulkanCommandBuffer =
    hadamard(bufferA, bufferB, bufferA)

fun Vulkan.hadamardAssign(bufferA: VulkanMatrixBuffer, bufferB: VulkanMatrixBuffer): VulkanCommandBuffer =
    hadamardAssign(bufferA.buffer, bufferB.buffer)
