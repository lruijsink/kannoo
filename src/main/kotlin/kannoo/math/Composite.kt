package kannoo.math

/**
 * Tensor composed of [slices] with [rank] N - 1. There are two types of composite tensor: [Matrix] and [NTensor]. The
 * only non-[Composite] tensor type is [Vector], and those together make up all possible [Tensor] types.
 *
 * @param T Composite tensor type
 *
 * @param S Slice tensor type
 */
sealed interface Composite<T : Tensor<T>, S : Tensor<S>> : Tensor<T> {

    /**
     * Slices that make up this tensor, themselves tensors of rank [rank]` - 1`
     */
    val slices: Array<S>

    /**
     * @param index Slice index to get
     *
     * @return Slice at index [index]
     */
    operator fun get(index: Int): S

    /**
     * @param index Slice index to set
     *
     * @param slice New slice value
     */
    operator fun set(index: Int, slice: S)
}
