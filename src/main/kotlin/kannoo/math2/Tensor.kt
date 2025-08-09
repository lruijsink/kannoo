package kannoo.math2

/**
 * A tensor is a numeric data structure of dimensions equal to its [rank] which abstracts the concept of [Vector]
 * (rank 1) and [Matrix] (rank 2) to higher dimensions. Tensors are regular and rectangular, each slice shares the same
 * dimensions. Tensors are always mutable to allow for more efficient operations, particularly those that modify the
 * vector in-place.
 *
 * Currently only 32-bit floating point ([Float]) arithmetic is supported.
 */
sealed interface Tensor {
    /**
     * The rank (dimensions) of this tensor. For example, a [Vector] is rank 1 and a [Matrix] is rank 2.
     */
    val rank: Int

    /**
     * The slices that this tensor is composed of, which are themselves tensors of rank [rank]` - 1`
     *
     * NOTE: This is an [Array] for memory efficiency reasons but it is NOT safe to write this to array!
     */
    val slices: Array<Tensor>

    /**
     * The tensor's size in its highest dimension. For example: a 3D tensor of 5 matrices has size 5, regardless of the
     * dimensions of those matrices.
     */
    val size: Int

    fun copy(): Tensor

    operator fun plus(t: Tensor): Tensor

    operator fun minus(t: Tensor): Tensor

    operator fun times(s: Float): Tensor

    operator fun div(s: Float): Tensor

    operator fun plusAssign(t: Tensor)

    operator fun minusAssign(t: Tensor)

    operator fun timesAssign(s: Float)

    operator fun divAssign(s: Float)

    fun map(transform: (Float) -> Float): Tensor

    fun mapAssign(transform: (Float) -> Float)
}

operator fun <T : Tensor> Float.times(tensor: T): T {
    @Suppress("UNCHECKED_CAST")
    return (tensor * this) as T
}

operator fun <T : Tensor> Number.times(tensor: T): T =
    tensor * this

operator fun <T : Tensor> T.times(s: Number): T {
    @Suppress("UNCHECKED_CAST")
    return (this * s.toFloat()) as T
}

operator fun <T : Tensor> T.timesAssign(s: Number) {
    timesAssign(s.toFloat())
}

operator fun <T : Tensor> T.div(s: Number): T {
    @Suppress("UNCHECKED_CAST")
    return (this / s.toFloat()) as T
}

operator fun <T : Tensor> T.divAssign(s: Number) {
    timesAssign(s.toFloat())
}
