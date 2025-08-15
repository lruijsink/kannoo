package kannoo.core

import kannoo.math.Tensor

class GradientReceiver(model: Model) {
    val gradients: Map<Tensor<*>, Tensor<*>> = model.layers
        .flatMap { it.learnable }
        .associateWith { param -> param.copyZero() }

    operator fun invoke(param: Tensor<*>, delta: Tensor<*>) {
        gradients[param]!! += delta // TODO: handle possible NPE better
    }

    fun apply(application: (param: Tensor<*>, gradient: Tensor<*>) -> Unit) {
        gradients.forEach { application(it.key, it.value) }
    }

    fun reset() {
        gradients.forEach { it.value.zero() }
    }

    // TODO: Make this a function on [Tensor]
    private fun Tensor<*>.copyZero(): Tensor<*> {
        val c = copy()
        c.zero()
        return c
    }
}
