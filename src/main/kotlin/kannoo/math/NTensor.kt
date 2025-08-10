package kannoo.math

class NTensor<S : Tensor<S>>(slices: List<S>) : Composite<NTensor<S>, S> {

    private val _slices = slices.toMutableList()
    override val slices = _slices

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

        return NTensor(size) { i -> this[i] + tensor[i] }
    }

    override operator fun minus(tensor: NTensor<S>): NTensor<S> {
        if (tensor.rank != this.rank || tensor.size != this.size)
            throw IllegalArgumentException("Incompatible dimensions")

        return NTensor(size) { i -> this[i] - tensor[i] }
    }

    override operator fun times(scalar: Float): NTensor<S> =
        NTensor(size) { i -> this[i] * scalar }

    override operator fun div(scalar: Float): NTensor<S> =
        NTensor(size) { i -> this[i] / scalar }

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
        NTensor(size) { i -> this[i].transform(function) }

    override fun assign(function: (Float) -> Float) {
        for (i in 0 until size) this[i].assign(function)
    }

    override fun copy(): NTensor<S> =
        NTensor(size) { i -> this[i].copy() }
}

inline fun <S : Tensor<S>> NTensor(size: Int, init: (index: Int) -> S): NTensor<S> =
    NTensor(List(size) { init(it) })

fun <S, T : Composite<T, S>> tensor(vararg tensors: T) =
    NTensor(tensors.toList())
