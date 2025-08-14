package kannoo.impl

import kannoo.core.ActivationFunction
import kannoo.core.InnerLayer
import kannoo.math.Matrix
import kannoo.math.Vector
import kannoo.math.emptyMatrix
import kannoo.math.randomMatrix

class DenseLayer(var weights: Matrix, var bias: Vector, activationFunction: ActivationFunction) :
    InnerLayer(bias.size, activationFunction) {

    constructor(size: Int, activationFunction: ActivationFunction) :
            this(weights = emptyMatrix(), bias = Vector(size), activationFunction = activationFunction)

    val initialized get() = weights.rows > 0

    override fun initialize(previousLayerSize: Int) {
        if (!initialized)
            weights = randomMatrix(size, previousLayerSize)
    }

    override fun forward(x: Vector): Vector {
        return activationFunction.compute(weights * x + bias)
    }

    override fun forward(x: Vector, update: (z: Vector, a: Vector) -> Unit) {
        val z = weights * x + bias
        update(z, activationFunction.compute(z))
    }

    override fun back(dz: Vector, x: Vector, update: (dW: Matrix, db: Vector) -> Unit): Vector {
        val da = dz * weights
        update(dz.outer(x), dz)
        return da
    }

    override fun backLast(dz: Vector, x: Vector, update: (dW: Matrix, db: Vector) -> Unit) {
        update(dz.outer(x), dz)
    }
}
