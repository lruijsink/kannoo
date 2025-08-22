
import kannoo.core.GradientReceiver
import kannoo.core.InputLayer
import kannoo.core.Model
import kannoo.core.Sample
import kannoo.impl.GrayscaleConvolutionLayer
import kannoo.impl.Logistic
import kannoo.impl.MeanSquaredError
import kannoo.impl.MiniBatchSGD
import kannoo.impl.ReLU
import kannoo.impl.grayscaleConvolutionLayer
import kannoo.io.readModelFromFile
import kannoo.io.writeModelToFile
import kannoo.math.Dimensions
import kannoo.math.Padding
import kannoo.math.ReflectionPadding
import kannoo.math.Shape
import kannoo.math.ZeroPadding
import kannoo.math.convolve
import kannoo.math.matrix
import kannoo.math.tensor
import kannoo.math.vector
import kotlin.system.measureTimeMillis

fun main() {
    val model = Model(
        InputLayer(Shape(6, 6)),
        grayscaleConvolutionLayer(
            kernelSize = Dimensions(3, 3),
            outputChannels = 1,
            activationFunction = Logistic,
        ),
    )
    val trainingSet = listOf(
        Sample(
            input = matrix(
                vector(0f, 0f, 0f, 1f, 1f, 1f),
                vector(0f, 0f, 0f, 1f, 1f, 1f),
                vector(0f, 0f, 0f, 1f, 1f, 1f),
                vector(1f, 1f, 1f, 0f, 0f, 0f),
                vector(1f, 1f, 1f, 0f, 0f, 0f),
                vector(1f, 1f, 1f, 0f, 0f, 0f),
            ),
            target = tensor(
                matrix(
                    vector(1f, 1f, 0f, 0f),
                    vector(1f, 1f, 0f, 0f),
                    vector(0f, 0f, 1f, 1f),
                    vector(0f, 0f, 1f, 1f),
                ),
            ),
        )
    )
    val sgd = MiniBatchSGD(model, MeanSquaredError, 0.1f, 1)
    val elapsed = measureTimeMillis {
        repeat(100000) {
            sgd.apply(trainingSet)
        }
    }
    println(elapsed)
}

fun convIO() {
    val model = Model(
        InputLayer(Shape(2, 2)),
        grayscaleConvolutionLayer(
            kernelSize = Dimensions(2, 2),
            padding = Padding(1, 1, ZeroPadding),
            stride = Dimensions(2, 2),
            outputChannels = 1,
            activationFunction = ReLU,
        ),
    )
    writeModelToFile(model, "convIOtest.kannoo")
    val modelFromDisk = readModelFromFile("convIOtest.kannoo")

    val original = model.layers[0]
    val fromDisk = modelFromDisk.layers[0]
    println(original)
    println(fromDisk)
}

fun fwd() {
    val X = matrix(
        vector(1f, 2f, 0f, 1f, 2f, 1f),
        vector(3f, 1f, 2f, 2f, 0f, 1f),
        vector(0f, 1f, 3f, 1f, 2f, 2f),
        vector(2f, 0f, 1f, 3f, 1f, 0f),
        vector(1f, 2f, 2f, 0f, 1f, 3f),
        vector(0f, 1f, 0f, 2f, 3f, 1f),
    )
    val K = matrix(
        vector(1f, -1f, 2f),
        vector(2f, 1f, -1f),
        vector(-1f, 2f, 1f),
    )
    println(convolve(X, K, Padding(1, 1, ReflectionPadding), Dimensions(2, 2)).prettyPrint())
    val gscl = GrayscaleConvolutionLayer(
        inputDimensions = Dimensions(X.rows, X.cols),
        kernelDimensions = Dimensions(K.rows, K.cols),
        padding = Padding(1, 1, ReflectionPadding),
        stride = Dimensions(2, 2),
        outputChannels = 1,
        activationFunction = ReLU,
    )
    gscl.kernels[0] = K
    println(gscl.preActivation(X)[0].prettyPrint())
}

fun gradients() {
    val X = matrix(
        vector(1f, 2f, 3f, 4f),
        vector(5f, 6f, 7f, 8f),
        vector(9f, 10f, 11f, 12f),
        vector(13f, 14f, 15f, 16f),
    )
    val K = matrix(
        vector(1f, 0f),
        vector(0f, -1f),
    )
    val gscl = GrayscaleConvolutionLayer(
        inputDimensions = Dimensions(X.rows, X.cols),
        kernelDimensions = Dimensions(K.rows, K.cols),
        padding = Padding(1, 1, ZeroPadding),
        stride = Dimensions(2, 2),
        outputChannels = 1,
        activationFunction = ReLU,
    )
    gscl.kernels[0] = K
    val model = Model(
        InputLayer(Shape(4, 4)),
        gscl,
    )
    val dY = matrix(
        vector(1f, 2f, 3f),
        vector(4f, 5f, 6f),
        vector(7f, 8f, 9f),
    )
    gscl.deltaInput(tensor(dY), X)
    gscl.gradients(tensor(dY), X, GradientReceiver(model))
}
