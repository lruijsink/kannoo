package kannoo

class Layer(
    val size: Int,
    val activationFunction: ActivationFunction = Linear,
) {
    val bias = Vector(size)
}
