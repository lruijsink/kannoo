package kannoo.math

/**
 * Bounded [Composite] with definite slice type [S], in the same way that [BoundedTensor] binds [Tensor].
 *
 * @param T Composite tensor type
 *
 * @param S Slice tensor type
 */
sealed interface BoundedComposite<T : BoundedTensor<T>, S : BoundedTensor<S>> : BoundedTensor<T>, Composite {

    /**
     * @param index Slice index to set
     *
     * @param slice New slice value
     */
    operator fun set(index: Int, slice: S)

    //
    // Default implementations and bindings:
    //

    override val slices: Array<S>

    override operator fun get(index: Int): S

    override fun set(index: Int, slice: Tensor) {
        @Suppress("UNCHECKED_CAST")
        set(index, slice as S)
    }
}
