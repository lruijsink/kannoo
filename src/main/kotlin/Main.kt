
import kannoo.example.rnd
import kannoo.impl.ReLU
import kannoo.math.Matrix
import kannoo.math.Shape
import kannoo.math.broadcastPlus
import kannoo.math.matrix
import kannoo.math.randomMatrix
import kannoo.math.randomVector
import kannoo.math.vector
import kannoo.vulkan.Vulkan
import kannoo.vulkan.VulkanDenseForward
import kannoo.vulkan.VulkanMatrixMul2
import kannoo.vulkan.VulkanModel
import kannoo.vulkan.createMatrixBuffer
import kannoo.vulkan.createShader
import kotlin.system.measureTimeMillis

fun testModel() {
    val vulkan = Vulkan()
    val model = VulkanModel(vulkan)

    fun printLayer(l: VulkanDenseForward) {
        println("Bias: ${l.bias.get()}")
        println("Weights:")
        println(l.weights.get().prettyPrint())
        println("Output:")
        println(l.output.get().prettyPrint())
        println()
    }

    fun printModel(m: VulkanModel) {
        //println(m.inputBuffer.get().prettyPrint())
        //println()
        printLayer(m.dense1)
        //printLayer(m.dense2)
        //println(m.outputBuffer.get().prettyPrint())
        //println()
    }

    model.inputBuffer.set(
        matrix(
            vector(-1f, 2f, -3f),
            vector( 4f, 5f, -6f),
        ),
    )
    model.dense1.bias.set(vector(1f, 2f, 3f, 4f))
    model.dense1.weights.set(
        matrix(
            vector( 9f,  8f,  7f),
            vector(-6f,  5f, -4f),
            vector( 3f, -2f,  1f),
            vector( 0f,  9f,  8f),
        ),
    )
    model.execute()

    println(
        model.inputBuffer.get()
            .multiplyTranspose(model.dense1.weights.get())
            .broadcastPlus(model.dense1.bias.get())
            .let { model.dense1.activation.compute(it) }
            .prettyPrint()
    )
    println()

    printModel(model)
}

fun main() {
    val rounds = 1000
    val cpuReduction = 100
    val cpuSkip = false
    val activation = ReLU

    val inputSize = 2048
    val outputSize = 1024
    val batchSize = 512
    val input = randomMatrix(batchSize, inputSize)
    val weights = randomMatrix(outputSize, inputSize)
    val bias = randomVector(outputSize)

    println("Creating Vulkan instance")
    val vulkan = Vulkan()
    println("Vulkan instance successfully created")

    println("Loading simple shader")
    val simpleShader = vulkan.createShader(
        fileName = "shaders/simple.spv",
        workgroupSize = 32,
    )
    println("Simple shader successfully loaded")

    println("Loading tiled shader")
    val tiledShader = vulkan.createShader(
        fileName = "shaders/tiled.spv",
        workgroupSize = 32,
    )
    println("Tiled shader successfully loaded")

    println("Loading tiled with constants shader")
    val tiledWithConstantsShader = vulkan.createShader(
        fileName = "shaders/tiled_with_constants.spv",
        workgroupSize = 32,
    )
    println("Tiled shader successfully loaded")

    println("Creating dense forward pipeline")
    val inputBuffer = vulkan.createMatrixBuffer(rows = batchSize, cols = inputSize)
    val vulkanDenseForward = VulkanDenseForward(
        vulkan = vulkan,
        input = inputBuffer,
        outputSize = outputSize,
        activation = activation,
    )
    println("Dense forward pipeline successfully created")

    println("Creating matrix mul 2 pipeline")
    val vulkanMatMul2 = VulkanMatrixMul2(
        vulkan = vulkan,
        input = vulkan.createMatrixBuffer(batchSize, inputSize),
        outputSize = outputSize,
        activation = activation,
    )
    println("Matrix mul 2 pipeline successfully created")

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

    val output = Matrix(batchSize, outputSize)
    val mmulsMs = measureTimeMillis {
        repeat(rounds) {
            vulkanMatMul2.runCommandBuffer()
        }
    }
    println("GPU took  ${rnd(mmulsMs / 1000.0f)} sec. for $rounds matrix muls (${rounds.toFloat() * 1000 / mmulsMs} ops/sec.)")
    println()

    if (!cpuSkip) {
        println("GPU is ${rnd((cpuMmMs.toFloat() / mmulsMs) * cpuReduction)} times faster at matrix muls")
        println()
    }

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

    val denseMs = measureTimeMillis {
        repeat(rounds) {
            vulkanDenseForward.runCommandBuffer()
        }
    }
    vulkanDenseForward.output.get()
    println("GPU took  ${rnd(denseMs / 1000.0f)} sec. for $rounds dense forwards (${rounds.toFloat() * 1000 / denseMs} ops/sec.)")
    println()

    if (!cpuSkip) {
        println("GPU is ${rnd((cpuDfMs.toFloat() / denseMs) * cpuReduction)} times faster at dense forwards")
        println()
    }

    println("Destroying Vulkan instance")
    vulkan.destroy()
    println("Vulkan instance successfully destroyed")
}
