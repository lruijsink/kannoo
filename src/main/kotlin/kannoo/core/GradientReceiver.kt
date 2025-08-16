package kannoo.core

import kannoo.math.TensorBase

class GradientReceiver(model: Model) {
    val gradients: Map<TensorBase, TensorBase> = model.layers
        .flatMap { it.learnable }
        .associateWith { param -> param.copyZero() }

    operator fun invoke(param: TensorBase, delta: TensorBase) {
        val gradient = gradients[param]
            ?: throw IllegalArgumentException("Can only apply gradients to params which are in InnerLayer.learnable")

        gradient += delta
    }

    fun apply(application: (param: TensorBase, gradient: TensorBase) -> Unit) {
        gradients.forEach { application(it.key, it.value) }
    }

    fun reset() {
        gradients.forEach { it.value.zeroAssign() }
    }
}
