package kannoo.vulkan

import kannoo.math.assertTensorEquals
import kannoo.math.randomIntMatrix
import org.junit.jupiter.api.Test

class VulkanMatMulAccTest {
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
        val factor = 2f
        transposeCases.forEach { (transposeA, transposeB) ->

            val matrixA = randomIntMatrix(outerDimA, innerDim)
                .let { if (transposeA) it.transpose() else it }

            val matrixB = randomIntMatrix(innerDim, outerDimB)
                .let { if (transposeB) it.transpose() else it }

            val matrixC = randomIntMatrix(outerDimA, outerDimB)

            val product = (if (transposeA) matrixA.transpose() else matrixA)
                .times((if (transposeB) matrixB.transpose() else matrixB))
                .times(factor)

            val matrixSum = matrixC + product

            println("A: ${if (transposeA) "(transposed)" else ""}")
            println(matrixA.prettyPrint())
            println()

            println("B: ${if (transposeB) "(transposed)" else ""}")
            println(matrixB.prettyPrint())
            println()

            println("C:")
            println(matrixC.prettyPrint())
            println()

            println("$factor * (A x B):")
            println(product.prettyPrint())
            println()

            println("C + $factor (A x B):")
            println(matrixSum.prettyPrint())
            println()

            val bufferA = vulkan.createMatrixBuffer(matrixA)
            val bufferB = vulkan.createMatrixBuffer(matrixB)
            val bufferC = vulkan.createMatrixBuffer(matrixC)

            val matMulCommandBuffer = vulkan.matMulAcc(bufferA, bufferB, bufferC, transposeA, transposeB)
            val execution = vulkan.createExecution(matMulCommandBuffer)

            matMulCommandBuffer.record(pushConstants(factor))
            execution.submit()

            println("C += $factor (A x B): (Vulkan)")
            println(bufferC.get().prettyPrint())
            println()

            assertTensorEquals(matrixSum, bufferC.get())

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
                // Powers of 2 won't cause rounding differences:
                val factor = listOf(0.125f, 0.25f, 0.5f, 1f, 2f, 4f, 8f).random()

                val outerDimA = (100..400).random()
                val outerDimB = (100..400).random()
                val innerDim = (100..400).random()
                val matrixA = randomIntMatrix(outerDimA, innerDim)
                    .let { if (transposeA) it.transpose() else it }

                val matrixB = randomIntMatrix(innerDim, outerDimB)
                    .let { if (transposeB) it.transpose() else it }

                val matrixC = randomIntMatrix(outerDimA, outerDimB)

                val product = (if (transposeA) matrixA.transpose() else matrixA)
                    .times((if (transposeB) matrixB.transpose() else matrixB))
                    .times(factor)

                val matrixSum = matrixC + product
                val bufferA = vulkan.createMatrixBuffer(matrixA)
                val bufferB = vulkan.createMatrixBuffer(matrixB)
                val bufferC = vulkan.createMatrixBuffer(matrixC)

                val matMulCommandBuffer = vulkan.matMulAcc(bufferA, bufferB, bufferC, transposeA, transposeB)
                val execution = vulkan.createExecution(matMulCommandBuffer)

                matMulCommandBuffer.record(pushConstants(factor))
                execution.submit()

                assertTensorEquals(matrixSum, bufferC.get())
                println("Correct for ${matrixA.shape} x ${matrixB.shape}")

            }
            println()
        }
    }
}
