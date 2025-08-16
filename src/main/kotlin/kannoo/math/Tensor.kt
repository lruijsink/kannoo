package kannoo.math

import kotlin.math.max
import kotlin.math.min

/**
 * A tensor is a numeric data structure of dimensions equal to its [rank] which abstracts the concept of [Vector]
 * (rank 1) and [Matrix] (rank 2) to higher dimensions. Tensors are regular and rectangular, each slice shares the same
 * dimensions.
 *
 * A rank 1 Tensor is a [Vector]:
 *
 * ```text
 * [1  2  3  4]
 * ```
 *
 * A rank 2 Tensor is a [Matrix], which is defined as a [Composite] tensor of [Vector] slices:
 *
 * ```text
 * [1  2]   <-- row vector: [1  2]
 * [3  4]   <-- row vector: [3  4]
 * ```
 *
 * A rank 3 Tensor is an [NTensor] with [Matrix] slices and can be imagined as a 3D rectangular prism:
 *
 * ```text
 * [ [1  2]   [5  6]
 *   [3  4]   [7  8] ]
 *    |        |
 *    |        '----> Second slice: [5  6]
 *    |                             [7  8]
 *    '----> First slice: [1  2]
 *                        [3  4]
 * ```
 *
 * A rank 4+ Tensor extends the same concept into higher dimensions.
 *
 * Instances of this class are mutable by default. Most operations have both a regular and in-place version, [map] and
 * [mapAssign] for example. The regular operation will allocate a new tensor, in-place versions will modify the existing
 * one. Be mindful about the sharing of references to tensors as this may lead to unexpected behavior. For example:
 *
 * ```kotlin
 * val v = vector(1f, 2f, 3f)
 * val vCopy = v
 *
 * v *= 100 // also reflects on vCopy, both share an underlying array
 * println(vCopy) // = [100, 200, 300] and NOT [1, 2, 3]
 * ```
 *
 * Tensors are limited to 32-bit floating point ([Float]) arithmetic.
 *
 * This interface provides a default implementation for many operations, [Vector] and [Matrix] largely override these in
 * favor of more efficient implementations.
 *
 * Writes to this class are NOT thread-safe, nor are any of its modifying operations.
 */
sealed interface Tensor<T : Tensor<T>> : TensorBase {

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
     * Casts any [TensorBase] to [T] as long as it has the same shape. This cast is guaranteed to be valid because every
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
    private fun castUnsafe(other: TensorBase): T {
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

    override operator fun plus(other: TensorBase): T =
        plus(castUnsafe(other))

    override operator fun minus(other: TensorBase) =
        minus(castUnsafe(other))

    override operator fun times(scalar: Float): T =
        map { it * scalar }

    override operator fun div(scalar: Float): T =
        map { it / scalar }

    override operator fun plusAssign(other: TensorBase) {
        plusAssign(castUnsafe(other))
    }

    override operator fun minusAssign(other: TensorBase) {
        minusAssign(castUnsafe(other))
    }

    override operator fun timesAssign(scalar: Float) {
        mapAssign { it * scalar }
    }

    override operator fun divAssign(scalar: Float) {
        mapAssign { it / scalar }
    }

    override fun map(function: (Float) -> Float): T

    override fun zip(other: TensorBase, combine: (left: Float, right: Float) -> Float): T =
        zip(castUnsafe(other), combine)

    override fun zipAssign(other: TensorBase, combine: (left: Float, right: Float) -> Float) {
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

    override infix fun hadamard(other: TensorBase): T =
        hadamard(castUnsafe(other))
}
