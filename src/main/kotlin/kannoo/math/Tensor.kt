package kannoo.math

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

    /**
     * @return A deep copy of this tensor, with equal rank, dimensions, and element values
     */
    fun copy(): Tensor

    /**
     * @param t The tensor to sum with, must be a tensor of equal rank and dimensions
     * @return A new vector `T` where `T[i]` = `this[i] + t[i]`
     * @throws UnsupportedTensorOperation if [t] is not a tensor of equal rank and dimensions
     */
    operator fun plus(t: Tensor): Tensor

    /**
     * @param t The tensor to subtract from this, must be a tensor of equal rank and dimensions
     * @return A new vector `T` where `T[i]` = `this[i] - t[i]`
     * @throws UnsupportedTensorOperation if [t] is not a tensor of equal rank and dimensions
     */
    operator fun minus(t: Tensor): Tensor

    /**
     * @param s Scalar value to multiply by
     * @return A new tensor `T` of equal rank and dimensions as this, where `T[i]` = `this[i] * s`
     */
    operator fun times(s: Float): Tensor

    /**
     * @param s Scalar value to divide by
     * @return A new tensor `T` of equal rank and dimensions as this, where `T[i]` = `this[i] / s`
     */
    operator fun div(s: Float): Tensor

    /**
     * Add each element in [t] to the corresponding element in this tensor, in-place.
     * @param t The tensor to sum with, must be a [Tensor] of equal rank and dimensions
     * @throws UnsupportedTensorOperation if [t] is not of equal rank or dimensions
     */
    operator fun plusAssign(t: Tensor)

    /**
     * Subtract each element in [t] from the corresponding element in this tensor, in-place.
     * @param t The tensor to subtract, must be a [Tensor] of equal rank and dimensions
     * @throws UnsupportedTensorOperation if [t] is not of equal rank or dimensions
     */
    operator fun minusAssign(t: Tensor)

    /**
     * Multiplies all values in this tensor by [s], in place.
     * @param s Scalar value to multiple by
     */
    operator fun timesAssign(s: Float)

    /**
     * Divides all values in this tensor by [s], in place.
     * @param s Scalar value to divide by
     */
    operator fun divAssign(s: Float)

    /**
     * Generic transform function which extends to all [Tensor] types. Use [transform] to preserve type information.
     * @param [function] Function to apply
     * @return A copy of this tensor with [function] applied to each scalar element
     */
    fun transform(function: (Float) -> Float): Tensor

    /**
     * Applies [transform] to all scalar elements in this tensor, in place.
     * @param [transform] Function to apply
     */
    fun reassign(transform: (Float) -> Float)
}

// TODO: doc
operator fun <T : Tensor> Float.times(tensor: T) =
    tensor.transformGeneric { it * this }


// TODO: doc
inline fun <V, T : Tensor> Iterable<V>.sumOfTensor(crossinline selector: (V) -> T): T {
    val itr = iterator()
    if (!itr.hasNext()) throw UnsupportedTensorOperation("Cannot sum over empty collection of tensors")

    val accumulator = selector(itr.next())
    while (itr.hasNext())
        accumulator += selector(itr.next())

    return accumulator
}

// TODO: doc
class UnsupportedTensorOperation(message: String) :
    IllegalArgumentException(message)
