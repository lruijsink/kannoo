package kannoo.core

import kannoo.math.Matrix
import kannoo.math.Vector

abstract class InnerLayer(
    val size: Int,
    val activationFunction: ActivationFunction,
) {
    abstract fun initialize(previousLayerSize: Int)

    abstract fun forward(x: Vector): Vector

    abstract fun forward(x: Vector, update: (z: Vector, a: Vector) -> Unit)

    abstract fun back(dz: Vector, x: Vector, update: (dW: Matrix, db: Vector) -> Unit): Vector

    abstract fun backLast(dz: Vector, x: Vector, update: (dW: Matrix, db: Vector) -> Unit)
}
