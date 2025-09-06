package kannoo.vulkan

import kannoo.core.ActivationFunction
import kannoo.impl.Linear
import kannoo.impl.Logistic
import kannoo.impl.ReLU

fun Vulkan.matMul(
    matrixA: VulkanMatrixBuffer,
    matrixB: VulkanMatrixBuffer,
    matrixOut: VulkanMatrixBuffer,
    transposeA: Boolean = false,
    transposeB: Boolean = false,
): VulkanCommandBuffer {
    val (m, n, k) = matMulDimsMNK(matrixA, matrixB, matrixOut, transposeA, transposeB)
    return createCommandBuffer(
        pipeline = createPipeline(
            shader = createShader("shaders/mat_mul.spv", 32),
            descriptorSet = createDescriptorSet(
                0 to matrixA.buffer,
                1 to matrixB.buffer,
                2 to matrixOut.buffer,
            ),
            specialization = createSpecialization(
                0 to m,
                1 to n,
                2 to k,
                3 to transposeA,
                4 to transposeB,
            ),
        ),
        groupCountX = matrixOut.cols.divCeil(32),
        groupCountY = matrixOut.rows.divCeil(32),
    )
}

fun Vulkan.matMulAcc(
    matrixA: VulkanMatrixBuffer,
    matrixB: VulkanMatrixBuffer,
    matrixOut: VulkanMatrixBuffer,
    factor: Float,
    transposeA: Boolean = false,
    transposeB: Boolean = false,
): VulkanCommandBuffer {
    val (m, n, k) = matMulDimsMNK(matrixA, matrixB, matrixOut, transposeA, transposeB)
    return createCommandBuffer(
        pipeline = createPipeline(
            shader = createShader("shaders/mat_mul_acc.spv", 32),
            descriptorSet = createDescriptorSet(
                0 to matrixA.buffer,
                1 to matrixB.buffer,
                2 to matrixOut.buffer,
            ),
            specialization = createSpecialization(
                0 to m,
                1 to n,
                2 to k,
                3 to transposeA,
                4 to transposeB,
            ),
            pushConstantCount = 1,
        ),
        groupCountX = matrixOut.cols.divCeil(32),
        groupCountY = matrixOut.rows.divCeil(32),
    ).record(pushConstants(factor))
}

fun Vulkan.matMulAcc(
    matrixA: VulkanMatrixBuffer,
    matrixB: VulkanMatrixBuffer,
    matrixOut: VulkanMatrixBuffer,
    transposeA: Boolean = false,
    transposeB: Boolean = false,
): VulkanCommandBuffer {
    val (m, n, k) = matMulDimsMNK(matrixA, matrixB, matrixOut, transposeA, transposeB)
    return createCommandBuffer(
        pipeline = createPipeline(
            shader = createShader("shaders/mat_mul_acc.spv", 32),
            descriptorSet = createDescriptorSet(
                0 to matrixA.buffer,
                1 to matrixB.buffer,
                2 to matrixOut.buffer,
            ),
            specialization = createSpecialization(
                0 to m,
                1 to n,
                2 to k,
                3 to transposeA,
                4 to transposeB,
            ),
            pushConstantCount = 1,
        ),
        groupCountX = matrixOut.cols.divCeil(32),
        groupCountY = matrixOut.rows.divCeil(32),
    )
}

fun Vulkan.matRowAcc(matrix: VulkanMatrixBuffer, vector: VulkanVectorBuffer, factor: Float) =
    createCommandBuffer(
        pipeline = createPipeline(
            shader = createShader("shaders/mat_row_acc.spv", 32),
            descriptorSet = createDescriptorSet(
                0 to matrix.buffer,
                1 to vector.buffer,
            ),
            specialization = createSpecialization(
                0 to matrix.cols,
                1 to matrix.rows,
            ),
            pushConstantCount = 1,
        ),
        groupCountX = vector.size.divCeil(32),
    ).record(pushConstants(factor))

fun Vulkan.matRowAcc(matrix: VulkanMatrixBuffer, vector: VulkanVectorBuffer) =
    createCommandBuffer(
        pipeline = createPipeline(
            shader = createShader("shaders/mat_row_acc.spv", 32),
            descriptorSet = createDescriptorSet(
                0 to matrix.buffer,
                1 to vector.buffer,
            ),
            specialization = createSpecialization(
                0 to matrix.cols,
                1 to matrix.rows,
            ),
            pushConstantCount = 1,
        ),
        groupCountX = vector.size.divCeil(32),
    )

fun Vulkan.matVecAdd(matrix: VulkanMatrixBuffer, vector: VulkanVectorBuffer) =
    createCommandBuffer(
        pipeline = createPipeline(
            shader = createShader("shaders/mat_vec_add.spv", 32),
            descriptorSet = createDescriptorSet(
                0 to matrix.buffer,
                1 to vector.buffer,
            ),
            specialization = createSpecialization(
                0 to matrix.cols,
                1 to matrix.rows,
            ),
        ),
        groupCountX = matrix.cols.divCeil(32),
        groupCountY = matrix.rows.divCeil(32),
    )

fun Vulkan.matActivation(matrix: VulkanMatrixBuffer, activation: ActivationFunction, derivative: Boolean = false) =
    createCommandBuffer(
        pipeline = createPipeline(
            shader = createShader("shaders/mat_elementwise_activation.spv", 32),
            descriptorSet = createDescriptorSet(
                0 to matrix.buffer,
            ),
            specialization = createSpecialization(
                0 to matrix.cols,
                1 to matrix.rows,
                2 to when (activation) {
                    is Linear -> 0
                    is ReLU -> 1
                    is Logistic -> 2
                    else -> throw IllegalStateException("Activation function $activation not yet supported")
                },
                3 to derivative,
            ),
        ),
        groupCountX = matrix.cols.divCeil(32),
        groupCountY = matrix.rows.divCeil(32),
    )

fun Vulkan.activate(buffer: VulkanBuffer, activation: ActivationFunction, derivative: Boolean = false) =
    createCommandBuffer(
        pipeline = createPipeline(
            shader = createShader("shaders/elementwise_activation.spv", 64),
            descriptorSet = createDescriptorSet(
                0 to buffer,
            ),
            specialization = createSpecialization(
                0 to (buffer.size / Float.SIZE_BYTES).toInt(),
                1 to activation.vulkanShaderId,
                2 to derivative,
            ),
        ),
        groupCountX = (buffer.size / Float.SIZE_BYTES).toInt().divCeil(64),
    )

fun matMulDimsMNK(
    matrixA: VulkanMatrixBuffer,
    matrixB: VulkanMatrixBuffer,
    matrixOut: VulkanMatrixBuffer,
    transposeA: Boolean = false,
    transposeB: Boolean = false,
): Triple<Int, Int, Int> {
    val innerA = if (transposeA) matrixA.rows else matrixA.cols
    val innerB = if (transposeB) matrixB.cols else matrixB.rows
    if (innerA != innerB)
        throw IllegalArgumentException("Cannot matrix-multiply A (inner size $innerA) with B (inner size $innerB)")

    val outerA = if (transposeA) matrixA.cols else matrixA.rows
    val outerB = if (transposeB) matrixB.rows else matrixB.cols
    if (matrixOut.rows != outerA || matrixOut.cols != outerB)
        throw IllegalArgumentException(
            "Matrix C (${matrixOut.rows} x ${matrixOut.cols}) cannot receive matrix-multiply product of " +
                    "A ($innerA x $outerA) and B ($outerB x $innerB)"
        )

    return Triple(
        matrixOut.rows, // M
        matrixOut.cols, // N
        innerA, // K
    )
}
