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
 */
sealed interface Tensor<T : Tensor<T>> : TensorBase {

    /**
     * The rank (dimensions) of this tensor. For example, a [Vector] is rank 1 and a [Matrix] is rank 2.
     */
    val rank: Int

    /**
     * The tensor's size in its highest dimension. For example: a 3D tensor of 5 matrices has size 5, regardless of the
     * dimensions of those matrices. Equivalent to [shape]`[0]`
     */
    val size: Int

    /**
     * The tensor's shape, from highest dimension to lowest. For example, the following tensor, with two 3 x 4 matrix
     * slices, has shape (2, 3, 4):
     *
     * ```text
     * [ [1  2  3  4]   [3  4  5  6]
     *   [5  6  7  8]   [7  8  9  0]
     *   [9  0  1  2]   [1  2  3  4] ]
     * ```
     */
    val shape: List<Int>

    /**
     * @return A deep copy of this tensor, with equal rank, dimensions, and element values
     */
    fun copy(): T

    /**
     * Generic (unsafe) overload of [plus]
     */
    operator fun plus(other: TensorBase): T =
        plus(castUnsafe(other))

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
     * Generic (unsafe) overload of [minus]
     */
    operator fun minus(other: TensorBase) =
        minus(castUnsafe(other))

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
     * @param scalar Scalar value to multiply by
     *
     * @return New tensor `T` where `T[i]` = `this[i] * scalar`
     */
    operator fun times(scalar: Float): T =
        map { it * scalar }

    /**
     * @param scalar Scalar value to divide by
     *
     * @return New tensor `T` where `T[i]` = `this[i] / scalar`
     */
    operator fun div(scalar: Float): T =
        map { it / scalar }

    /**
     * Generic (unsafe) overload of [plusAssign]
     */
    operator fun plusAssign(other: TensorBase) {
        plusAssign(castUnsafe(other))
    }

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
     * Generic (unsafe) overload of [minusAssign]
     */
    operator fun minusAssign(other: TensorBase) {
        minusAssign(castUnsafe(other))
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
     * Multiplies all values in this tensor by [scalar], in-place.
     *
     * @param scalar Scalar value to multiply by
     */
    operator fun timesAssign(scalar: Float) {
        mapAssign { it * scalar }
    }

    /**
     * Divides all values in this tensor by [scalar], in-place.
     *
     * @param scalar Scalar value to divide by
     */
    operator fun divAssign(scalar: Float) {
        mapAssign { it / scalar }
    }

    /**
     * @param [function] Function to apply
     *
     * @return A new tensor with `function` applied elementwise to this tensor
     */
    fun map(function: (Float) -> Float): T

    /**
     * Applies [function] to all scalar elements in this tensor, in-place.
     *
     * @param [function] Function to apply
     */
    fun mapAssign(function: (Float) -> Float)

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
     * Generic (unsafe) overload of [zip]
     */
    fun zip(other: TensorBase, combine: (left: Float, right: Float) -> Float): T =
        zip(castUnsafe(other), combine)

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
     * Generic (unsafe) overload of [zipAssign]
     */
    fun zipAssign(other: TensorBase, combine: (left: Float, right: Float) -> Float) {
        zipAssign(castUnsafe(other), combine)
    }

    /**
     * Calls [operation] with each element in the tensor, recursively from highest to lowest rank. For example, given
     * the following 3D tensor:
     *
     * ```text
     * [ [1  2]   [5  6]
     *   [3  4]   [7  8] ]
     * ```
     *
     * The call order would be (1, 2, 3, 4, 5, 6, 7, 8)
     *
     * @param operation Function to call
     */
    fun forEachElement(operation: (Float) -> Unit)

    /**
     * Reduces all elements in the tensor in recursive order, from highest to lowest rank, same as [forEachElement],
     * and returns the accumulated result. Requires a non-empty tensor.
     *
     * @param operation Reducing operation to apply
     *
     * @return Accumulated value after reduction
     *
     * @throws TensorOperationException if this tensor is empty
     */
    fun reduce(operation: (accumulator: Float, element: Float) -> Float): Float {
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

    /**
     * @return The sum over all elements in this tensor.
     */
    fun sum(): Float =
        reduce { x, y -> x + y }

    /**
     * @return The minimum (smallest) element in this tensor.
     */
    fun min(): Float =
        reduce(::min)

    /**
     * @return The maximum (largest) element in this tensor.
     */
    fun max(): Float =
        reduce(::max)

    /**
     * Sets all elements in the tensor to zero (`0f`)
     */
    fun zero() {
        mapAssign { 0f }
    }

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
     * Generic (unsafe) overload of [hadamard]
     */
    infix fun hadamard(other: TensorBase): T =
        hadamard(castUnsafe(other))

    /**
     * Casts any [TensorBase] to [T] as long as it has the same shape.
     *
     * TODO: Explain this better
     */
    private fun castUnsafe(other: TensorBase): T {
        if (this.shape != (other as Tensor<*>).shape)
            throw IncompatibleShapeException("Cannot combine tensors with different shapes")

        @Suppress("UNCHECKED_CAST")
        return other as T
    }
}

/**
 * [Tensor.times] with the arguments flipped. These operation are symmetric and equivalent.
 *
 * @param other Tensor to multiply by `this`
 *
 * @return New tensor `T` where `T[i]` = `other[i] * this`
 */
operator fun <T : Tensor<T>> Float.times(other: T) =
    other.map { it * this }

/**
 * Sums over an [Iterable] with elements of type [V] which get mapped to tensors of type [T], in the same way as
 * with [Iterable.sumOf].
 *
 * @param V Iterator element type
 *
 * @param T Tensor type
 *
 * @param selector Function that selects a [T] tensor from a [V] element
 *
 * @return The selected [T] tensors summed with [Tensor.plus]
 *
 * @throws EmptyTensorException if the selected tensors do not all share the same shape
 */
inline fun <V, T : Tensor<T>> Iterable<V>.sumOfTensor(crossinline selector: (V) -> T): T {
    val itr = iterator()
    if (!itr.hasNext())
        throw EmptyTensorException("Cannot sum over empty collection of tensors")

    val accumulator = selector(itr.next())
    while (itr.hasNext())
        accumulator += selector(itr.next())

    return accumulator
}

/**
 * Base class for all illegal or unsupported tensor operation exceptions.
 */
abstract class TensorOperationException(message: String) : IllegalArgumentException(message)

/**
 * Thrown by operations that require both tensors to have the same shape (see [Tensor.shape]).
 */
open class IncompatibleShapeException(message: String) : TensorOperationException(message)

/**
 * Thrown by operations that require a non-empty tensor. In general, tensors should not be empty.
 */
open class EmptyTensorException(message: String) : TensorOperationException(message)
