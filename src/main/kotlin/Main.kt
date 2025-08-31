
import kannoo.example.rnd
import kannoo.math.Matrix
import kannoo.math.Shape
import kannoo.math.randomMatrix
import kannoo.vulkan.Shader
import kannoo.vulkan.Vulkan
import kannoo.vulkan.VulkanMatrixMultiply
import kotlin.system.measureTimeMillis

fun main() {
    val R = 1000
    val inputSize = 2048
    val outputSize = 1024
    val batchSize = 512
    val input = randomMatrix(batchSize, inputSize)
    val weights = randomMatrix(outputSize, inputSize)

    val simpleShader = Shader(
        fileName = "shaders/simple.spv",
        workgroupSize = 32,
    )

    val tiledShader = Shader(
        fileName = "shaders/tiled.spv",
        workgroupSize = 32,
    )

    println("Creating Vulkan instance")
    val vulkan = Vulkan()
    println("Vulkan instance successfully created")

    println("Creating matrix multiply pipeline")
    val vulkanMatrixMultiply = VulkanMatrixMultiply(
        vulkan = vulkan,
        shader = tiledShader,
        inputSize = inputSize,
        outputSize = outputSize,
        batchSize = batchSize,
    )
    println("Matrix multiply pipeline successfully created")

    println()
    println("                         batches   input/output vectors")
    println("                              |     |")
    println("                              v     v")
    println("Input matrix dimensions:    ${input.shape}")
    println("Weights matrix dimensions: ${weights.shape}")
    println("Output matrix dimensions:   ${Shape(batchSize, outputSize)} = Input * Weights.transpose()")
    println()
    val calc = R.toLong() * batchSize * inputSize * outputSize
    println("Total work = $R repetitions * $batchSize batches * $inputSize input vectors * $outputSize weights per input")
    println("           = $calc (${rnd(calc / 1_000_000_000_000.0f)} trillion) calculations")
    println()

    var ref: Matrix? = null
    val cpuMs = measureTimeMillis {
        repeat(R / 100) {
            ref = Matrix(Array(batchSize) { i -> weights * input[i] })
        }
    }
    ref!!
    println("CPU took ${rnd(cpuMs / 1000.0f)} sec. for ${R / 100}   matrix multiplications (${R.toFloat() * 10 / cpuMs} mmuls/sec.)")
    println()

    val output = Matrix(batchSize, outputSize)
    val runsMs = measureTimeMillis {
        repeat(R) {
            vulkanMatrixMultiply.multiplyTransposed(input, weights, output)
        }
    }
    println("GPU took  ${rnd(runsMs / 1000.0f)} sec. for $R matrix multiplications (${R.toFloat() * 1000 / runsMs} mmuls/sec.)")
    println()
    println("GPU is ${rnd((cpuMs.toFloat() / runsMs) * 100)} times faster")
    println()

    println("Destroying Vulkan instance")
    vulkanMatrixMultiply.destroy()
    vulkan.destroy()
    println("Vulkan instance successfully destroyed")
}
