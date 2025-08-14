package kannoo.core

import kannoo.impl.Softmax
import kannoo.math.Matrix
import kannoo.math.Vector

class BackPropagator(
    val model: Model,
    val cost: CostFunction,
) {
    val z = model.layers.map { Vector(it.size) }.toMutableList()
    val a = model.layers.map { Vector(it.size) }.toMutableList()

    inline fun calculatePartials(sample: Sample, crossinline update: (i: Int, dW: Matrix, db: Vector) -> Unit) {
        // Forward pass:
        var o = sample.input
        model.layers.forEachIndexed { i, layer ->
            layer.forward(o) { z, a ->
                this.z[i] = z
                this.a[i] = a
                o = this.a[i]
            }
        }

        // Backward pass:
        var da = cost.derivative(sample.target, a[model.layers.size - 1])
        for (i in (model.layers.size - 1) downTo 0) {
            val dz =
                if (model.layers[i].activationFunction == Softmax) da // Combined into one operation
                else da.hadamard(model.layers[i].activationFunction.derivative(z[i]))

            if (i > 0)
                da = model.layers[i].back(dz = dz, x = a[i - 1]) { dW, db -> update(i, dW, db) }
            else
                model.layers[i].backLast(dz = dz, x = sample.input) { dW, db -> update(i, dW, db) }
        }
    }
}
