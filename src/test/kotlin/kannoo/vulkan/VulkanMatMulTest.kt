package kannoo.vulkan

import kannoo.math.assertTensorEquals
import kannoo.math.randomIntMatrix
import org.junit.jupiter.api.Test

class VulkanMatMulTest {
    private val vulkan = Vulkan()

    private val transposeCases = listOf(
        Pair(false, false),
        Pair(false, true),
        Pair(true, false),
        Pair(true, true),
    )

    @Test
    fun `Test small matrix multiplication with Vulkan`() {
        val outerDimA = 3
        val outerDimB = 5
        val innerDim = 4
        transposeCases.forEach { (transposeA, transposeB) ->

            val matrixA = randomIntMatrix(outerDimA, innerDim)
                .let { if (transposeA) it.transpose() else it }

            val matrixB = randomIntMatrix(innerDim, outerDimB)
                .let { if (transposeB) it.transpose() else it }

            val matrixC = (if (transposeA) matrixA.transpose() else matrixA)
                .times((if (transposeB) matrixB.transpose() else matrixB))

            println("A: ${if (transposeA) "(transposed)" else ""}")
            println(matrixA.prettyPrint())
            println()

            println("B: ${if (transposeB) "(transposed)" else ""}")
            println(matrixB.prettyPrint())
            println()

            println("C = A x B:")
            println(matrixC.prettyPrint())
            println()

            val bufferA = vulkan.createMatrixBuffer(matrixA)
            val bufferB = vulkan.createMatrixBuffer(matrixB)
            val bufferC = vulkan.createMatrixBuffer(rows = outerDimA, cols = outerDimB)

            val matMulCommandBuffer = vulkan.matMul(bufferA, bufferB, bufferC, transposeA, transposeB)
            val execution = vulkan.createExecution(matMulCommandBuffer)
            execution.submit()

            println("C = A x B: (Vulkan)")
            println(bufferC.get().prettyPrint())
            println()

            assertTensorEquals(matrixC, bufferC.get())

            println()
            println()
        }
    }

    @Test
    fun `Test large matrix multiplication with Vulkan`() {
        transposeCases.forEach { (transposeA, transposeB) ->
            print(if (transposeA) "A.transpose" else "A")
            print(" x ")
            print(if (transposeB) "B.transpose" else "B")
            println(":")
            repeat(10) {
                val outerDimA = (100..400).random()
                val outerDimB = (100..400).random()
                val innerDim = (100..400).random()

                val matrixA = randomIntMatrix(outerDimA, innerDim)
                    .let { if (transposeA) it.transpose() else it }

                val matrixB = randomIntMatrix(innerDim, outerDimB)
                    .let { if (transposeB) it.transpose() else it }

                val matrixC = (if (transposeA) matrixA.transpose() else matrixA)
                    .times((if (transposeB) matrixB.transpose() else matrixB))

                val bufferA = vulkan.createMatrixBuffer(matrixA)
                val bufferB = vulkan.createMatrixBuffer(matrixB)
                val bufferC = vulkan.createMatrixBuffer(rows = outerDimA, cols = outerDimB)

                val matMulCommandBuffer = vulkan.matMul(bufferA, bufferB, bufferC, transposeA, transposeB)
                val execution = vulkan.createExecution(matMulCommandBuffer)
                execution.submit()

                assertTensorEquals(matrixC, bufferC.get())
                println("Correct for ${matrixA.shape} x ${matrixB.shape}")

            }
            println()
        }
    }
}
