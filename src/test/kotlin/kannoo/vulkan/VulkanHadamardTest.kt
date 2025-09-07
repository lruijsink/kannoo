package kannoo.vulkan

import kannoo.math.assertTensorEquals
import kannoo.math.randomIntMatrix
import org.junit.jupiter.api.Test

class VulkanHadamardTest {
    val vulkan = Vulkan()

    @Test
    fun `Test hadamard product`() {
        val matrixA = randomIntMatrix(3, 5)
        val matrixB = randomIntMatrix(3, 5)
        val product = matrixA hadamard matrixB

        println("A:")
        println(matrixA.prettyPrint())
        println()

        println("B:")
        println(matrixB.prettyPrint())
        println()

        println("A hadamard B:")
        println(product.prettyPrint())
        println()

        val bufferA = vulkan.createMatrixBuffer(matrixA)
        val bufferB = vulkan.createMatrixBuffer(matrixB)
        val bufferOut = vulkan.createMatrixBuffer(matrixA.rows, matrixA.cols)

        val hadamardCommandBuffer = vulkan.hadamard(bufferA, bufferB, bufferOut)
        val execution = vulkan.createExecution(hadamardCommandBuffer)
        execution.submit()

        println("A hadamard B (Vulkan):")
        println(bufferOut.get().prettyPrint())
        println()

        assertTensorEquals(product, bufferOut.get())
    }

    @Test
    fun `Test hadamard assign product`() {
        val matrixA = randomIntMatrix(3, 5)
        val matrixB = randomIntMatrix(3, 5)
        val product = matrixA hadamard matrixB

        println("A:")
        println(matrixA.prettyPrint())
        println()

        println("B:")
        println(matrixB.prettyPrint())
        println()

        println("A hadamard B:")
        println(product.prettyPrint())
        println()

        val bufferA = vulkan.createMatrixBuffer(matrixA)
        val bufferB = vulkan.createMatrixBuffer(matrixB)

        val hadamardCommandBuffer = vulkan.hadamardAssign(bufferA, bufferB)
        val execution = vulkan.createExecution(hadamardCommandBuffer)
        execution.submit()

        println("A hadamard B (Vulkan):")
        println(bufferA.get().prettyPrint())
        println()

        assertTensorEquals(product, bufferA.get())
    }
}
