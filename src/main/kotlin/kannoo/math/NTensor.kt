package kannoo.math

@JvmInline
value class NTensor<S : Tensor<S>>(override val slices: Array<S>) : Composite<NTensor<S>, S> {

    override val rank: Int get() = slices[0].rank + 1

    override val size: Int get() = slices.size

    val depth: Int get() = slices[0].size

    override operator fun get(index: Int): S =
        slices[index]

    override operator fun set(index: Int, slice: S) {
        if (slice.size != depth)
            throw IllegalArgumentException("Incorrect slice size") // TODO

        slices[index] = slice
    }

    override operator fun plus(tensor: NTensor<S>): NTensor<S> {
        if (tensor.rank != this.rank || tensor.size != this.size)
            throw IllegalArgumentException("Incompatible dimensions")

        return transformIndexed { i -> this[i] + tensor[i] }
    }

    override operator fun minus(tensor: NTensor<S>): NTensor<S> {
        if (tensor.rank != this.rank || tensor.size != this.size)
            throw IllegalArgumentException("Incompatible dimensions")

        return transformIndexed { i -> this[i] - tensor[i] }
    }

    override operator fun times(scalar: Float): NTensor<S> =
        transformIndexed { i -> this[i] * scalar }

    override operator fun div(scalar: Float): NTensor<S> =
        transformIndexed { i -> this[i] / scalar }

    override operator fun plusAssign(tensor: NTensor<S>) {
        if (tensor.size != this.size) throw IllegalArgumentException("Incompatible")
        for (i in 0 until size) this[i].plusAssign(tensor[i])
    }

    override operator fun minusAssign(tensor: NTensor<S>) {
        if (tensor.size != this.size) throw IllegalArgumentException("Incompatible")
        for (i in 0 until size) this[i].minusAssign(tensor[i])
    }

    override operator fun timesAssign(scalar: Float) {
        assign { it * scalar }
    }

    override operator fun divAssign(scalar: Float) {
        assign { it / scalar }
    }

    override fun transform(function: (Float) -> Float): NTensor<S> =
        transformIndexed { i -> this[i].transform(function) }

    override fun assign(function: (Float) -> Float) {
        for (i in 0 until size) this[i].assign(function)
    }

    override fun copy(): NTensor<S> {
        val slicesCopy = slices.copyOf()
        for (i in 0 until size)
            slicesCopy[i] = slices[i].copy()
        return NTensor(slicesCopy)
    }

    override fun zip(other: NTensor<S>, combine: (Float, Float) -> Float): NTensor<S> =
        transformIndexed { i -> this[i].zip(other[i], combine) }

    override fun zipAssign(other: NTensor<S>, combine: (Float, Float) -> Float) {
        for (i in 0 until size)
            this[i].zipAssign(other[i], combine)
    }

    private inline fun assignIndexed(crossinline function: (Int) -> S) {
        for (i in 0 until size)
            this[i] = function(i)
    }

    private inline fun transformIndexed(crossinline function: (Int) -> S): NTensor<S> {
        val res = copy()
        res.assignIndexed(function)
        return res
    }
}

fun <S, T : Composite<T, S>> tensor(vararg tensors: T): NTensor<T> {
    @Suppress("KotlinConstantConditions")
    return NTensor(tensors as Array<T>)
}
