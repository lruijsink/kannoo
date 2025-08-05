package kannoo.old

import kannoo.core.ActivationFunction
import kannoo.impl.Linear
import kannoo.math.Vector

class Layer(
    val size: Int,
    val activationFunction: ActivationFunction = Linear,
) {
    val bias = Vector(size)
}
