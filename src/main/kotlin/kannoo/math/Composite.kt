package kannoo.math

sealed interface Composite<T : Tensor<T>, S : Tensor<S>> : Tensor<T> {

    /**
     * The slices that this tensor is composed of, which are themselves tensors of rank [rank]` - 1`
     *
     * NOTE: This is an [Array] for memory efficiency reasons and therefore writeable.
     */
    val slices: Array<S>

    override val rank: Int
        get() = slices[0].rank + 1

    override val size: Int
        get() = slices.size

    operator fun get(index: Int): S =
        slices[index]

    operator fun set(index: Int, slice: S) {
        slices[index] = slice
    }
}
