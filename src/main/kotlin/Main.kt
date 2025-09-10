import kannoo.core.Model
import kannoo.core.Sample
import kannoo.core.inputLayer
import kannoo.example.rnd
import kannoo.impl.Logistic
import kannoo.impl.MeanSquaredError
import kannoo.impl.MiniBatchSGD
import kannoo.impl.ReLU
import kannoo.impl.denseLayer
import kannoo.math.Matrix
import kannoo.math.Shape
import kannoo.math.Vector
import kannoo.math.matrix
import kannoo.math.randomMatrix
import kannoo.math.randomVector
import kannoo.math.vector
import kannoo.vulkan.ActivationMode.DERIVATIVE
import kannoo.vulkan.DenseConfig
import kannoo.vulkan.Vulkan
import kannoo.vulkan.VulkanDense
import kannoo.vulkan.VulkanModel
import kannoo.vulkan.activate
import kannoo.vulkan.createExecution
import kannoo.vulkan.createMatrixBuffer
import kannoo.vulkan.createVectorBuffer
import kannoo.vulkan.hadamardAssign
import kannoo.vulkan.matMul
import kannoo.vulkan.matMulAcc
import kannoo.vulkan.matRowAcc
import kannoo.vulkan.matVecAdd
import kannoo.vulkan.pushConstants
import kotlin.system.measureTimeMillis

val vulkan = Vulkan()

val x = matrix(
    vector(0f, 0f),
    vector(1f, 0f),
    vector(0f, 1f),
    vector(1f, 1f),
)

val t = matrix(
    vector(1f, 0f, 0f, 0f),
    vector(0f, 1f, 0f, 1f),
    vector(0f, 1f, 0f, 1f),
    vector(1f, 0f, 1f, 1f),
)

fun testModelPerf() {
    val batchSize = 640
    val model = VulkanModel(
        vulkan,
        inputSize = 28 * 28,
        batchSize = batchSize,
        configs = listOf(
            DenseConfig(32, Logistic),
            DenseConfig(10, Logistic),
        )
    )
    val x = randomMatrix(batchSize, 28 * 28)
    val t = randomMatrix(batchSize, 10)
    repeat(100) {
        val ms = measureTimeMillis {
            repeat(60_000 / batchSize) {
                model.backProp(x, t, -0.1f)
            }
        }
        println("Took $ms ms")
    }
}

fun testModel() {
    val model = VulkanModel(
        vulkan,
        inputSize = x.cols,
        batchSize = x.rows,
        configs = listOf(
            DenseConfig(8, Logistic),
            DenseConfig(t.cols, Logistic),
        )
    )

    val ms = measureTimeMillis {
        repeat(1000) {
            model.backProp(x, t, -0.5f)
        }
    }

    println("GPU took $ms ms")
    println(model.compute(x).prettyPrint())
    println()
}

fun testModelOld() {
    val m = Model(
        inputLayer(2),
        denseLayer(8, Logistic),
        denseLayer(4, Logistic),
    )
    val sgd = MiniBatchSGD(m, MeanSquaredError, 0.5f, 4)
    val s = List(x.rows) { i -> Sample(x[i], t[i]) }
    val ms = measureTimeMillis {
        repeat(1000) {
            sgd.trainBatch(s)
        }
    }
    println("CPU took $ms ms")
    println((m.compute(x) as Matrix).prettyPrint())
    println()
}

fun testModelManual() {

    val x0 = vulkan.createMatrixBuffer(
        matrix(
            vector(0f, 0f),
            vector(1f, 0f),
            vector(0f, 1f),
            vector(1f, 1f),
        )
    )

    val t = matrix(
        vector(1f, 0f, 0f, 0f),
        vector(0f, 1f, 0f, 1f),
        vector(0f, 1f, 0f, 1f),
        vector(1f, 0f, 1f, 1f),
    )

    val batchSize = x0.rows
    val outputSize = t.cols
    val hiddenSize = 3

    val w0 = vulkan.createMatrixBuffer(randomMatrix(hiddenSize, x0.cols))
    val b0 = vulkan.createVectorBuffer(Vector(hiddenSize))
    val z0 = vulkan.createMatrixBuffer(batchSize, hiddenSize)
    val f0 = Logistic

    val x1 = vulkan.createMatrixBuffer(z0.rows, z0.cols)
    val w1 = vulkan.createMatrixBuffer(randomMatrix(outputSize, hiddenSize))
    val b1 = vulkan.createVectorBuffer(Vector(outputSize))
    val z1 = vulkan.createMatrixBuffer(batchSize, outputSize)
    val f1 = Logistic

    val y = vulkan.createMatrixBuffer(z1.rows, z1.cols)

    val fwd = vulkan.createExecution(
        vulkan.matMul(x0, w0, z0, transposeB = true),
        vulkan.matVecAdd(z0, b0),
        vulkan.activate(z0, x1, f0),

        vulkan.matMul(x1, w1, z1, transposeB = true),
        vulkan.matVecAdd(z1, b1),
        vulkan.activate(z1, y, f1),
    )

    val C = MeanSquaredError
    val r = 0.1f

    val dy = vulkan.createMatrixBuffer(y.rows, y.cols)

    val dz1 = vulkan.createMatrixBuffer(z1.rows, z1.cols)
    val dx1 = vulkan.createMatrixBuffer(x1.rows, x1.cols)

    val dz0 = vulkan.createMatrixBuffer(z0.rows, z0.cols)
    // unnecessary: val dx0 = vulkan.createMatrixBuffer(x0.rows, x0.cols)

    val back = vulkan.createExecution(
        vulkan.activate(z1, dz1, f0, DERIVATIVE),
        vulkan.hadamardAssign(dz1, dy),
        vulkan.matMul(dz1, w1, dx1),
        vulkan.matMulAcc(dz1, x1, w1, transposeA = true).record(pushConstants(-r)),
        vulkan.matRowAcc(dz1, b1).record(pushConstants(-r)),

        vulkan.activate(z0, dz0, f0, DERIVATIVE),
        vulkan.hadamardAssign(dz0, dx1),
        // unnecessary: vulkan.matMul(dz0, w0, dx0),
        vulkan.matMulAcc(dz0, x0, w0, transposeA = true).record(pushConstants(-r)),
        vulkan.matRowAcc(dz0, b0).record(pushConstants(-r)),
    )

    repeat(10000) {
        fwd.submit()
        dy.set(C.derivative(t, y.get()))
        back.submit()
    }

    println(y.get().prettyPrint())
    println()
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

    val execDense2 = vulkan.createExecution(vulkanDense2.forward)
    val msDense2 = measureTimeMillis {
        repeat(rounds) {
            execDense2.submit()
        }
    }
    println("GPU took  ${rnd(msDense2 / 1000.0f)} sec. for $rounds dense 2s (${rounds.toFloat() * 1000 / msDense2} ops/sec.)")
    println()

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
    testModelPerf()
}
