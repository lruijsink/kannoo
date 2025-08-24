package kannoo.core

import kannoo.math.Tensor
import java.util.IdentityHashMap

class GradientAccumulator(model: Model) : GradientReceiver {
    val gradients: Map<Tensor, Tensor>

    init {
        val gs = IdentityHashMap<Tensor, Tensor>() // Index by reference, not value
        for (layer in model.layers)
            for (tensor in layer.learnable)
                gs[tensor] = tensor.copyZero()

        gradients = gs
    }

    override operator fun invoke(param: Tensor, delta: Tensor) {
        val gradient = gradients[param]
            ?: throw IllegalArgumentException("Can only apply gradients to params which are in InnerLayer.learnable")

        gradient += delta
    }

    fun reset() {
        gradients.forEach { it.value.zeroAssign() }
    }
}
