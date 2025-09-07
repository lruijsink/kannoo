package kannoo.vulkan

import kannoo.impl.Logistic
import kannoo.math.assertTensorEquals
import kannoo.math.matrix
import kannoo.math.vector
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test

class VulkanDenseTest {
    val vulkan = Vulkan()

    val input = matrix(
        vector(1f, 2f, 3f),
        vector(4f, 5f, 6f),
    )

    val weights = matrix(
        vector(0.1f, 0.2f, 0.3f),
        vector(0.4f, 0.5f, 0.6f),
    )

    val bias = vector(0.5f, -0.5f)

    val preActivation = matrix(
        vector(1.9f, 2.7f),
        vector(3.7f, 7.2f),
    )

    val layer = VulkanDense(
        vulkan = vulkan,
        input = vulkan.createMatrixBuffer(input),
        output = vulkan.createMatrixBuffer(preActivation.rows, preActivation.cols),
        deltaInput = vulkan.createMatrixBuffer(input.rows, input.cols), // unused
        deltaOutput = vulkan.createMatrixBuffer(preActivation.rows, preActivation.cols), // unused
        activation = Logistic,
    )

    @BeforeEach
    fun setup() {
        layer.weights.set(weights)
        layer.bias.set(bias)
    }

    @Test
    fun `Test forward pass on a small layer`() {
        vulkan.createExecution(layer.forward).submit()
        assertTensorEquals(preActivation, layer.preActivation.get())
        assertTensorEquals(Logistic.compute(preActivation), layer.output.get())
    }

    @Test
    fun `Test manual backprop pass on a small layer`() {
        val deltaPreActivation = matrix(
            vector(0.1f, -0.2f),
            vector(0.05f, 0.02f),
        )

        val dZ = vulkan.createMatrixBuffer(deltaPreActivation)
        val dW = vulkan.createMatrixBuffer(weights.copyZero())
        val dB = vulkan.createVectorBuffer(bias.copyZero())
        val dX = vulkan.createMatrixBuffer(input.copyZero())

        val exec = vulkan.createExecution(
            vulkan.matMulAcc(dZ, layer.input, dW, transposeA = true).record(pushConstants(1f)),
            vulkan.matRowAcc(dZ, dB).record(pushConstants(1f)),
            vulkan.matMul(dZ, layer.weights, dX),
        )

        exec.submit()

        assertTensorEquals(
            matrix(
                vector(0.3f, 0.45f, 0.6f),
                vector(-0.12f, -0.3f, -0.48f),
            ),
            dW.get(),
        )

        assertTensorEquals(
            vector(0.15f, -0.18f),
            dB.get(),
        )

        assertTensorEquals(
            matrix(
                vector(-0.07f, -0.08f, -0.09f),
                vector(0.013f, 0.02f, 0.027f),
            ),
            dX.get(),
        )
    }
}
