package kannoo.math

@JvmInline
value class CompositeTensor(override val slices: Array<Tensor>) : Tensor {

    override val rank: Int get() = slices[0].rank + 1

    override val size: Int get() = slices.size

    val depth: Int get() = slices[0].size

    operator fun get(index: Int): Tensor =
        slices[index]

    operator fun set(index: Int, tensor: Tensor) {
        if (tensor.size != depth)
            throw IllegalArgumentException("Incorrect slice size") // TODO

        slices[index] = tensor
    }

    override operator fun plus(tensor: Tensor): CompositeTensor {
        if (tensor !is CompositeTensor || tensor.rank != this.rank || tensor.size != this.size)
            throw IllegalArgumentException("Incompatible dimensions")

        return CompositeTensor(size) { i -> this[i] + tensor[i] }
    }

    override operator fun minus(tensor: Tensor): CompositeTensor {
        if (tensor !is CompositeTensor || tensor.rank != this.rank || tensor.size != this.size)
            throw IllegalArgumentException("Incompatible dimensions")

        return CompositeTensor(size) { i -> this[i] - tensor[i] }
    }

    override operator fun times(scalar: Float): CompositeTensor =
        CompositeTensor(size) { i -> this[i] * scalar }

    override operator fun div(scalar: Float): CompositeTensor =
        CompositeTensor(size) { i -> this[i] / scalar }

    override operator fun plusAssign(tensor: Tensor) {
        if (tensor !is Matrix || tensor.size != this.size) throw IllegalArgumentException("Incompatible")
        for (i in 0 until size) this[i].plusAssign(tensor[i])
    }

    override operator fun minusAssign(tensor: Tensor) {
        if (tensor !is Matrix || tensor.size != this.size) throw IllegalArgumentException("Incompatible")
        for (i in 0 until size) this[i].minusAssign(tensor[i])
    }

    override operator fun timesAssign(scalar: Float) {
        reassign { it * scalar }
    }

    override operator fun divAssign(scalar: Float) {
        reassign { it / scalar }
    }

    override fun transform(function: (Float) -> Float): CompositeTensor =
        CompositeTensor(size) { i -> this[i].transform(function) }

    override fun reassign(function: (Float) -> Float) {
        for (i in 0 until size) this[i].reassign(function)
    }

    override fun copy(): CompositeTensor =
        CompositeTensor(size) { i -> this[i].copy() }
}

inline fun CompositeTensor(size: Int, init: (index: Int) -> Tensor): CompositeTensor =
    CompositeTensor(Array(size) { init(it) })

fun composite(vararg tensors: Tensor): CompositeTensor {
    @Suppress("KotlinConstantConditions")
    return CompositeTensor(tensors as Array<Tensor>)
}
