package kannoo.math2

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

    override operator fun plus(t: Tensor): CompositeTensor {
        if (t !is CompositeTensor || t.rank != this.rank || t.size != this.size)
            throw IllegalArgumentException("Incompatible dimensions")

        return CompositeTensor(size) { i -> this[i] + t[i] }
    }

    override operator fun minus(t: Tensor): CompositeTensor {
        if (t !is CompositeTensor || t.rank != this.rank || t.size != this.size)
            throw IllegalArgumentException("Incompatible dimensions")

        return CompositeTensor(size) { i -> this[i] - t[i] }
    }

    override operator fun times(s: Float): CompositeTensor =
        CompositeTensor(size) { i -> this[i] * s }

    override operator fun div(s: Float): CompositeTensor =
        CompositeTensor(size) { i -> this[i] / s }

    override operator fun plusAssign(t: Tensor) {
        if (t !is Matrix || t.size != this.size) throw IllegalArgumentException("Incompatible")
        for (i in 0 until size) this[i].plusAssign(t[i])
    }

    override operator fun minusAssign(t: Tensor) {
        if (t !is Matrix || t.size != this.size) throw IllegalArgumentException("Incompatible")
        for (i in 0 until size) this[i].minusAssign(t[i])
    }

    override operator fun timesAssign(s: Float) {
        reassign { it * s }
    }

    override operator fun divAssign(s: Float) {
        reassign { it / s }
    }

    override fun transform(function: (Float) -> Float): CompositeTensor =
        CompositeTensor(size) { i -> this[i].transform(function) }

    override fun reassign(transform: (Float) -> Float) {
        for (i in 0 until size) this[i].reassign(transform)
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
