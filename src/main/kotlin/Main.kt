
import kannoo.example.rnd
import kannoo.impl.Logistic
import kannoo.impl.ReLU
import kannoo.math.Matrix
import kannoo.math.Shape
import kannoo.math.broadcastPlus
import kannoo.math.matrix
import kannoo.math.randomMatrix
import kannoo.math.randomVector
import kannoo.math.vector
import kannoo.vulkan.Vulkan
import kannoo.vulkan.VulkanDense
import kannoo.vulkan.createExecution
import kannoo.vulkan.createMatrixBuffer
import kannoo.vulkan.matActivation
import kannoo.vulkan.matMul
import kannoo.vulkan.matMulAcc
import kannoo.vulkan.pushConstants
import kotlin.system.measureTimeMillis

val vulkan = Vulkan()

fun testModel() {
}

fun testMatMulAcc() {
    val s = 0.5f

    val A = matrix(
        vector(1f, 2f, 3f),
        vector(4f, 5f, 6f),
    )

    val B = matrix(
        vector(-1f, 0.5f, 1f, -0.5f),
        vector(1f, -1f, -0.5f, 0.5f),
        vector(-0.5f, 1f, 0.5f, -1f),
    ).transpose()

    val C = Matrix(A.rows, B.rows)

    C += (A multiplyTranspose B) * s
    println(C.prettyPrint())
    println()

    C += (A multiplyTranspose  B) * s
    println(C.prettyPrint())
    println()


    val bA = vulkan.createMatrixBuffer(A)
    val bB = vulkan.createMatrixBuffer(B)
    val bC = vulkan.createMatrixBuffer(C.copyZero())
    val cmd = vulkan.matMulAcc(bA, bB, bC, transposeB = true)
    val matTransposeMulAcc = vulkan.createExecution(cmd)

    println()
    println(bC.get().prettyPrint())

    println()
    cmd.record(pushConstants(s))
    matTransposeMulAcc.submit()
    println(bC.get().prettyPrint())

    println()
    cmd.record(pushConstants(3 * s))
    matTransposeMulAcc.submit()
    println(bC.get().prettyPrint())
}

fun testMatMul() {
    val a1 = matrix(
        vector(1f, 2f),
    )

    val b1 = matrix(
        vector(1f, 0f),
        vector(0f, 1f),
    )

    val a2 = matrix(
        vector(1f, 2f, 3f),
        vector(4f, 5f, 6f),
    )

    val b2 = matrix(
        vector(1f, -2f, 3f, -4f),
        vector(-5f, 6f, -7f, 8f),
        vector(9f, -10f, 11f, -12f),
    )

    val a = a2
    val b = b2.transpose()
    val c = a * b.transpose()

    println(c.prettyPrint())
    println()

    val aB = vulkan.createMatrixBuffer(a)
    val bB = vulkan.createMatrixBuffer(b)
    val cB = vulkan.createMatrixBuffer(c.copyZero())

    vulkan.createExecution(vulkan.matMul(aB, bB, cB, transposeA = false, transposeB = true)).submit()

    println(cB.get().prettyPrint())
}

fun testActivation() {
    val a = Logistic
    val m = matrix(vector(-1f, -2f, 3f), vector(-4f, 5f, 6f))
    println(a.compute(m).prettyPrint())
    println()

    val b = vulkan.createMatrixBuffer(m)
    vulkan.createExecution(vulkan.matActivation(b, a, derivative = false)).submit()
    println(b.get().prettyPrint())
}

fun testPerf() {
    val rounds = 1000
    val cpuReduction = 100
    val cpuSkip = true
    val activation = ReLU

    val inputSize = 1024
    val outputSize = 1024
    val batchSize = 1024
    val input = randomMatrix(batchSize, inputSize)
    val weights = randomMatrix(outputSize, inputSize)
    val bias = randomVector(outputSize)

    println("Creating dense 2 pipeline")
    val vulkanDense2 = VulkanDense(
        vulkan = vulkan,
        input = vulkan.createMatrixBuffer(batchSize, inputSize),
        output = vulkan.createMatrixBuffer(batchSize, outputSize),
        deltaInput = vulkan.createMatrixBuffer(batchSize, inputSize),
        deltaOutput = vulkan.createMatrixBuffer(batchSize, outputSize),
        activation = activation,
    )
    println("Dense 2 pipeline successfully created")

    println()
    println("                         batches   input/output vectors")
    println("                              |     |")
    println("                              v     v")
    println("Input matrix dimensions:    ${input.shape}")
    println("Weights matrix dimensions: ${weights.shape}")
    println("Output matrix dimensions:   ${Shape(batchSize, outputSize)} = Input * Weights.transpose()")
    println()
    val calc = rounds.toLong() * batchSize * inputSize * outputSize
    println("Total work = $rounds repetitions * $batchSize batches * $inputSize input vectors * $outputSize weights per input")
    println("           = $calc (${rnd(calc / 1_000_000_000_000.0f)} trillion) calculations")
    println()

    var cpuDfMs = 1L
    if (!cpuSkip) {
        cpuDfMs = measureTimeMillis {
            repeat(rounds / cpuReduction) {
                activation.compute(input.multiplyTranspose(weights).broadcastPlus(bias))
            }
        }
        println("CPU took ${rnd(cpuDfMs / 1000.0f)} sec. for ${rounds / cpuReduction}   dense forwards (${rounds.toFloat() * 1000 / (cpuDfMs * cpuReduction)} ops/sec.)")
        println()
    }

    var cpuMmMs = 1L
    if (!cpuSkip) {
        cpuMmMs = measureTimeMillis {
            repeat(rounds / cpuReduction) {
                input.multiplyTranspose(weights)
            }
        }
        println("CPU took ${rnd(cpuMmMs / 1000.0f)} sec. for ${rounds / cpuReduction}   matrix muls (${rounds.toFloat() * 1000 / (cpuMmMs * cpuReduction)} ops/sec.)")
        println()
    }

    val execDense2 = vulkan.createExecution(vulkanDense2.forward)
    val msDense2 = measureTimeMillis {
        repeat(rounds) {
            execDense2.submit()
        }
    }
    println("GPU took  ${rnd(msDense2 / 1000.0f)} sec. for $rounds dense 2s (${rounds.toFloat() * 1000 / msDense2} ops/sec.)")
    println()

    if (!cpuSkip) {
        println("GPU is ${rnd((cpuMmMs.toFloat() / msDense2) * cpuReduction)} times faster at dense 2")
        println()
    }

    val execDense2BackProp = vulkan.createExecution(vulkanDense2.backProp)
    val msDenseBackProp2 = measureTimeMillis {
        repeat(rounds) {
            vulkanDense2.recordBackProp(0.1f)
            execDense2BackProp.submit()
        }
    }
    println("GPU took  ${rnd(msDenseBackProp2 / 1000.0f)} sec. for $rounds dense back prop 2s (${rounds.toFloat() * 1000 / msDenseBackProp2} ops/sec.)")
    println()

    println("Destroying Vulkan instance")
    vulkan.destroy()
    println("Vulkan instance successfully destroyed")
}

fun main() {
    testPerf()
}
