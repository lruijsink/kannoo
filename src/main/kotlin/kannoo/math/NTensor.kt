package kannoo.math

@JvmInline
value class NTensor<S : Tensor<S>>(override val slices: Array<S>) : Composite<NTensor<S>, S> {

    override fun map(function: (Float) -> Float): NTensor<S> =
        transformIndexed { i -> this[i].map(function) }

    override fun mapAssign(function: (Float) -> Float) {
        for (i in 0 until size) this[i].mapAssign(function)
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
