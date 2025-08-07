package kannoo.io

import kannoo.core.ActivationFunction
import kannoo.impl.LeakyReLU
import kannoo.impl.Linear
import kannoo.impl.Logistic
import kannoo.impl.ReLU
import kannoo.impl.Softmax

fun deserializeActivationFunction(value: String): ActivationFunction =
    when (value) {
        "ReLU" -> ReLU
        "LeakyReLU" -> LeakyReLU
        "Logistic" -> Logistic
        "Linear" -> Linear
        "Softmax" -> Softmax
        else -> throw IllegalArgumentException("Unknown activation function '$value")
    }

fun ActivationFunction.serialize(): String =
    this::class.simpleName!!
