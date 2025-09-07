package kannoo.vulkan

import kannoo.math.assertTensorEquals
import kannoo.math.broadcastPlus
import kannoo.math.randomIntMatrix
import kannoo.math.randomIntVector
import org.junit.jupiter.api.Test

class VulkanMatVecAddTest {
    private val vulkan = Vulkan()

    @Test
    fun `Test vector broadcast addition over a small matrix with Vulkan`() {
        val cols = 5
        val matrix = randomIntMatrix(4, cols)
        val vector = randomIntVector(cols)
        val broadcast = matrix broadcastPlus vector

        println("Matrix:")
        println(matrix.prettyPrint())
        println()

        println("Vector:")
        println(vector.prettyPrint())
        println()

        println("Matrix + Vector (broadcast):")
        println(broadcast.prettyPrint())
        println()

        val matrixBuffer = vulkan.createMatrixBuffer(matrix)
        val vectorBuffer = vulkan.createVectorBuffer(vector)
        val matRowAccCommandBuffer = vulkan.matVecAdd(matrixBuffer, vectorBuffer)
        val execution = vulkan.createExecution(matRowAccCommandBuffer)
        execution.submit()

        println("Matrix + Vector (broadcast) (Vulkan):")
        println(matrixBuffer.get().prettyPrint())

        assertTensorEquals(broadcast, matrixBuffer.get())
    }
}
