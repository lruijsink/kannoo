package kannoo.math

import kotlin.math.max
import kotlin.math.min

/**
 * Bounded interface that all [Tensor] implementations should implement, to allow for operations like `a + b` to be
 * statically typed; accepting and returning concrete types, such as [Vector], rather than the supertype [Tensor].
 *
 * This interface:
 * - Overrides every operation which returns [Tensor] to specify it returns [T] instead
 * - Overload every operation which accepts [Tensor] arguments to additionally accept [T]
 *
 * For example:
 *
 * ```kotlin
 * // Tensor defines:
 * fun plus (other: Tensor): Tensor
 *
 * // BoundedTensor<T> specializes this to:
 * override fun plus (other: Tensor): T // casts `other` to `T`
 * fun plus (other: T): T // no cast, implicitly type-safe
 * ```
 */
sealed interface BoundedTensor<T : BoundedTensor<T>> : Tensor {

    //
    // Extensions specific to Tensor<T>:
    //

    /**
     * @param other Tensor to sum with
     *
     * @return New tensor `T` where `T[i]` = `this[i] + other[i]`
     *
     * @throws IncompatibleShapeException if the tensors do not have the same [shape]
     */
    operator fun plus(other: T): T =
        zip(other) { x, y -> x + y }

    /**
     * @param other Tensor to subtract
     *
     * @return New tensor `T` where `T[i]` = `this[i] - other[i]`
     *
     * @throws IncompatibleShapeException if the tensors do not have the same [shape]
     */
    operator fun minus(other: T): T =
        zip(other) { x, y -> x - y }

    /**
     * Add each element in [other] to the corresponding element in this tensor, in-place.
     *
     * @param other Tensor to sum with
     *
     * @throws IncompatibleShapeException if the tensors do not have the same [shape]
     */
    operator fun plusAssign(other: T) {
        zipAssign(other) { x, y -> x + y }
    }

    /**
     * Subtract each element in [other] from the corresponding element in this tensor, in-place.
     *
     * @param other Tensor to subtract
     *
     * @throws IncompatibleShapeException if the tensors do not have the same [shape]
     */
    operator fun minusAssign(other: T) {
        zipAssign(other) { x, y -> x - y }
    }

    /**
     * Zips this tensor with another and applies the combining operation over each (element, element) pair in both
     * tensors.
     *
     * @param other Tensor to zip with
     *
     * @param combine Operation to apply to each zipped (element, element) pair
     *
     * @return New tensor `T` where for each element `i`:
     *
     * `T[i]` = `combine(this[i], other[i])`
     *
     * @throws IncompatibleShapeException if the tensors do not have the same [shape]
     */
    fun zip(other: T, combine: (left: Float, right: Float) -> Float): T

    /**
     * Zips this tensor with another and applies the combining operation over each (element, element) pair in-place,
     * such that for each element `i`:
     *
     * `this[i]` = `combine(this[i], other[i])`
     *
     * @param other Tensor to zip with
     *
     * @param combine Operation to apply to each zipped (element, element) pair
     *
     * @throws IncompatibleShapeException if the tensors do not have the same [shape]
     */
    fun zipAssign(other: T, combine: (left: Float, right: Float) -> Float)

    /**
     * Computes the Hadamard product, which multiples each element in both tensors elementwise.
     *
     * @param other Tensor to compute the product of
     *
     * @return New tensor `T` where for each element `i`:
     *
     * `T[i]` = `this[i] * other[i]`
     *
     * @throws IncompatibleShapeException if the tensors do not have the same [shape]
     */
    infix fun hadamard(other: T): T =
        zip(other) { x, y -> x * y }

    /**
     * Casts any [Tensor] to [T] as long as it has the same shape. This cast is guaranteed to be valid because every
     * possible rank of tensor always corresponds to the same class:
     *
     * 1 = [Vector], 2 = [Matrix], 3+ = [NTensor]
     *
     * @param other Tensor to cast to [T]
     *
     * @return [other] cast to [T]
     *
     * @throws IncompatibleGenericTensorException If the tensors do not have the same shape
     */
    private fun castUnsafe(other: Tensor): T {
        if (this.shape != other.shape)
            throw IncompatibleGenericTensorException("Cannot cast a generic tensor to a tensor with different shape")

        @Suppress("UNCHECKED_CAST")
        return other as T
    }

    //
    // Default implementations and specializations of [TensorBase] as [T]:
    //

    override val totalElements: Int get() =
        shape.dimensions.reduce { x, y -> x * y }

    override fun copy(): T

    override fun copyZero(): T {
        val copy = copy()
        copy.zeroAssign()
        return copy
    }

    override operator fun unaryMinus(): T =
        map { -it }

    override operator fun plus(other: Tensor): T =
        plus(castUnsafe(other))

    override operator fun minus(other: Tensor) =
        minus(castUnsafe(other))

    override operator fun times(scalar: Float): T =
        map { it * scalar }

    override operator fun div(scalar: Float): T =
        map { it / scalar }

    override operator fun plusAssign(other: Tensor) {
        plusAssign(castUnsafe(other))
    }

    override operator fun minusAssign(other: Tensor) {
        minusAssign(castUnsafe(other))
    }

    override operator fun timesAssign(scalar: Float) {
        mapAssign { it * scalar }
    }

    override operator fun divAssign(scalar: Float) {
        mapAssign { it / scalar }
    }

    override fun map(function: (Float) -> Float): T

    override fun zip(other: Tensor, combine: (left: Float, right: Float) -> Float): T =
        zip(castUnsafe(other), combine)

    override fun zipAssign(other: Tensor, combine: (left: Float, right: Float) -> Float) {
        zipAssign(castUnsafe(other), combine)
    }

    override fun reduce(operation: (accumulator: Float, element: Float) -> Float): Float {
        var first = true
        var acc = 0f

        forEachElement { element ->
            acc = if (first) element else operation(acc, element)
            first = false
        }

        if (first)
            throw EmptyTensorException("Cannot reduce an empty tensor")

        return acc
    }

    override fun sum(): Float =
        reduce { x, y -> x + y }

    override fun min(): Float =
        reduce(::min)

    override fun max(): Float =
        reduce(::max)

    override fun zeroAssign() {
        mapAssign { 0f }
    }

    override infix fun hadamard(other: Tensor): T =
        hadamard(castUnsafe(other))
}
