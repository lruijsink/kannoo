package kannoo.core

import kannoo.math.Tensor
import java.util.IdentityHashMap

class GradientReceiver(model: Model) {
    val gradients: Map<Tensor, Tensor>

    init {
        val gs = IdentityHashMap<Tensor, Tensor>() // Index by reference, not value
        for (layer in model.layers)
            for (tensor in layer.learnable)
                gs[tensor] = tensor.copyZero()

        gradients = gs
    }

    operator fun invoke(param: Tensor, delta: Tensor) {
        val gradient = gradients[param]
            ?: throw IllegalArgumentException("Can only apply gradients to params which are in InnerLayer.learnable")

        gradient += delta
    }

    fun apply(application: (param: Tensor, gradient: Tensor) -> Unit) {
        gradients.forEach { application(it.key, it.value) }
    }

    fun reset() {
        gradients.forEach { it.value.zeroAssign() }
    }
}
