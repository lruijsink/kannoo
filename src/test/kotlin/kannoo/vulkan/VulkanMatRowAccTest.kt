package kannoo.vulkan

import kannoo.math.assertTensorEquals
import kannoo.math.randomIntMatrix
import kannoo.math.randomIntVector
import org.junit.jupiter.api.Test

class VulkanMatRowAccTest {
    private val vulkan = Vulkan()

    @Test
    fun `Test row summation over small matrix with Vulkan`() {
        val cols = 5
        val factor = 1.5f
        val matrix = randomIntMatrix(4, cols)
        val vector = randomIntVector(cols)
        val sumRowsAcc = vector + matrix.sumRows() * factor

        println("Matrix:")
        println(matrix.prettyPrint())
        println()

        println("Vector:")
        println(vector.prettyPrint())
        println()

        println("Vector + $factor * sumRows(Matrix):")
        println(sumRowsAcc.prettyPrint())
        println()

        val matrixBuffer = vulkan.createMatrixBuffer(matrix)
        val vectorBuffer = vulkan.createVectorBuffer(vector)
        val matRowAccCommandBuffer = vulkan.matRowAcc(matrixBuffer, vectorBuffer)
        val execution = vulkan.createExecution(matRowAccCommandBuffer)

        matRowAccCommandBuffer.record(pushConstants(factor))
        execution.submit()

        println("Vector + $factor * sumRows(Matrix) (Vulkan):")
        println(vectorBuffer.get().prettyPrint())

        assertTensorEquals(sumRowsAcc, vectorBuffer.get())
    }

    @Test
    fun `Test row summation over large matrix with Vulkan`() {
        repeat(10) {
            val cols = (100..1000).random()
            val factor = listOf(0.125f, 0.25f, 0.5f, 1f, 1.5f, 2f, 4f, 8f).random()

            val matrix = randomIntMatrix((100..1000).random(), cols, -1000..1000)
            val vector = randomIntVector(cols, -1000..1000)
            val sumRowsAcc = vector + matrix.sumRows() * factor

            val matrixBuffer = vulkan.createMatrixBuffer(matrix)
            val vectorBuffer = vulkan.createVectorBuffer(vector)
            val matRowAccCommandBuffer = vulkan.matRowAcc(matrixBuffer, vectorBuffer)
            val execution = vulkan.createExecution(matRowAccCommandBuffer)

            matRowAccCommandBuffer.record(pushConstants(factor))
            execution.submit()

            assertTensorEquals(sumRowsAcc, vectorBuffer.get())
            println("Correct for size $cols vector, over ${matrix.rows} rows, with factor $factor")
        }
    }
}
