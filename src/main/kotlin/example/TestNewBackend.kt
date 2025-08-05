package example

import kannoo.impl.Logistic
import kannoo.impl.MeanSquaredError
import kannoo.impl.DenseLayer
import kannoo.core.InputLayer
import kannoo.impl.MiniBatchSGD
import kannoo.core.Model
import kannoo.core.TrainingExample
import kannoo.math.vectorOf
import kotlin.math.roundToInt

fun testNewBackend() {
    val model = Model(InputLayer(4), DenseLayer(3, Logistic), DenseLayer(1, Logistic))
    val sgd = MiniBatchSGD(model, MeanSquaredError, 1, 0.1)
    val trainingData = listOf(
        TrainingExample(vectorOf(0.0, 0.0, 0.0, 0.0), vectorOf(0.0)),
        TrainingExample(vectorOf(0.0, 0.0, 1.0, 0.0), vectorOf(0.0)),
        TrainingExample(vectorOf(0.0, 0.0, 0.0, 1.0), vectorOf(0.0)),
        TrainingExample(vectorOf(0.0, 0.0, 1.0, 1.0), vectorOf(1.0)),
        TrainingExample(vectorOf(0.0, 1.0, 0.0, 0.0), vectorOf(0.0)),
        TrainingExample(vectorOf(0.0, 1.0, 1.0, 0.0), vectorOf(1.0)),
        TrainingExample(vectorOf(0.0, 1.0, 0.0, 1.0), vectorOf(1.0)),
        TrainingExample(vectorOf(0.0, 1.0, 1.0, 1.0), vectorOf(1.0)),
        TrainingExample(vectorOf(1.0, 0.0, 0.0, 0.0), vectorOf(0.0)),
        TrainingExample(vectorOf(1.0, 0.0, 1.0, 0.0), vectorOf(1.0)),
        TrainingExample(vectorOf(1.0, 0.0, 0.0, 1.0), vectorOf(1.0)),
        TrainingExample(vectorOf(1.0, 0.0, 1.0, 1.0), vectorOf(0.0)),
        TrainingExample(vectorOf(1.0, 1.0, 0.0, 0.0), vectorOf(1.0)),
        TrainingExample(vectorOf(1.0, 1.0, 1.0, 0.0), vectorOf(0.0)),
        TrainingExample(vectorOf(1.0, 1.0, 0.0, 1.0), vectorOf(0.0)),
        TrainingExample(vectorOf(1.0, 1.0, 1.0, 1.0), vectorOf(1.0)),
    )
    repeat(10) { i ->
        repeat(1000) { sgd.apply(trainingData) }
        println("Epoch ${(i + 1) * 1000}:")
        trainingData.forEach {
            println(it.input.scalars.map(Double::roundToInt).toString() + " -> " + model.compute(it.input).scalars.map(::rnd))
        }
        println()
    }
}
