package kannoo.vulkan

import kannoo.impl.Linear
import kannoo.impl.Logistic
import kannoo.impl.ReLU
import kannoo.math.assertTensorEquals
import kannoo.math.randomIntMatrix
import kannoo.vulkan.ActivationMode.DERIVATIVE
import org.junit.jupiter.api.Test

class VulkanActivationTest {
    val vulkan = Vulkan()

    val supportedActivations = listOf(
        Linear,
        ReLU,
        Logistic,
    )

    @Test
    fun `Test activation of a small buffer with Vulkan`() {
        for (fn in supportedActivations) {
            val name = fn::class.simpleName

            val matrix = randomIntMatrix(5, 6)
            val input = vulkan.createMatrixBuffer(matrix)
            val output = vulkan.createMatrixBuffer(matrix.copyZero())

            val activated = fn.compute(matrix)
            val derivative = fn.derivative(matrix)

            println("Input:")
            println(matrix.prettyPrint())
            println()

            println("Activated ($name):")
            println(activated.prettyPrint())
            println()

            println("Derivative ($name):")
            println(derivative.prettyPrint())
            println()

            val activateCommandBuffer = vulkan.activate(input, output, fn)
            val activateExecution = vulkan.createExecution(activateCommandBuffer)
            activateExecution.submit()

            println("Activated ($name) (Vulkan):")
            println(output.get().prettyPrint())
            println()

            assertTensorEquals(activated, output.get())

            val derivativeCommandBuffer = vulkan.activate(input, output, fn, DERIVATIVE)
            val derivativeExecution = vulkan.createExecution(derivativeCommandBuffer)
            derivativeExecution.submit()

            println("Derivative ($name) (Vulkan):")
            println(output.get().prettyPrint())
            println()

            assertTensorEquals(derivative, output.get())

            println()
            println()
        }
    }

    @Test
    fun `Test activation-assign of a small buffer with Vulkan`() {
        for (fn in supportedActivations) {
            val name = fn::class.simpleName

            val matrix = randomIntMatrix(5, 6)
            val buffer = vulkan.createMatrixBuffer(matrix)

            val activated = fn.compute(matrix)
            val derivative = fn.derivative(matrix)

            println("Input:")
            println(matrix.prettyPrint())
            println()

            println("Activated ($name):")
            println(activated.prettyPrint())
            println()

            println("Derivative ($name):")
            println(derivative.prettyPrint())
            println()

            val activateCommandBuffer = vulkan.activateAssign(buffer, fn)
            val activateExecution = vulkan.createExecution(activateCommandBuffer)
            activateExecution.submit()

            println("Activated ($name) (Vulkan):")
            println(buffer.get().prettyPrint())
            println()

            assertTensorEquals(activated, buffer.get())

            buffer.set(matrix)
            val derivativeCommandBuffer = vulkan.activateAssign(buffer, fn, DERIVATIVE)
            val derivativeExecution = vulkan.createExecution(derivativeCommandBuffer)
            derivativeExecution.submit()

            println("Derivative ($name) (Vulkan):")
            println(buffer.get().prettyPrint())
            println()

            assertTensorEquals(derivative, buffer.get())

            println()
            println()
        }
    }
}
