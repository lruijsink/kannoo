package kannoo.core

import kannoo.math.Tensor

class GradientReceiver(model: Model) {
    val gradients: Map<Tensor, Tensor> = model.layers
        .flatMap { it.learnable }
        .associateWith { param -> param.copyZero() }

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
