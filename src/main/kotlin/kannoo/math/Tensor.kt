package kannoo.math

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
 * Writes to a tensor object are NOT thread-safe, nor are any of its modifying operations.
 */
sealed interface Tensor {

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
    val shape: Shape

    /**
     * Total number of elements in the tensor, across all ranks. Equivalent to multiplying the size of each slice. For
     * example: a tensor of 3 matrix slices with dimensions 2 x 5 has 3 x 2 x 5 = 30 total elements.
     */
    val totalElements: Int

    /**
     * @return A deep copy of this tensor, with equal shape and element values
     */
    fun copy(): Tensor

    /**
     * @return A zeroed-out copy of this tensor, with equal shape.
     */
    fun copyZero(): Tensor

    /**
     * @return New tensor `T` where `T[i]` = `-this[i]`
     */
    operator fun unaryMinus(): Tensor

    /**
     * @param other Tensor to sum with
     *
     * @return New tensor `T` where `T[i]` = `this[i] + other[i]`
     *
     * @throws IncompatibleShapeException if the tensors do not have the same [shape]
     */
    operator fun plus(other: Tensor): Tensor

    /**
     * @param other Tensor to subtract
     *
     * @return New tensor `T` where `T[i]` = `this[i] - other[i]`
     *
     * @throws IncompatibleShapeException if the tensors do not have the same [shape]
     */
    operator fun minus(other: Tensor): Tensor

    /**
     * @param scalar Scalar value to multiply by
     *
     * @return New tensor `T` where `T[i]` = `this[i] * scalar`
     */
    operator fun times(scalar: Float): Tensor

    /**
     * @param scalar Scalar value to divide by
     *
     * @return New tensor `T` where `T[i]` = `this[i] / scalar`
     */
    operator fun div(scalar: Float): Tensor

    /**
     * Add each element in [other] to the corresponding element in this tensor, in-place.
     *
     * @param other Tensor to sum with
     *
     * @throws IncompatibleShapeException if the tensors do not have the same [shape]
     */
    operator fun plusAssign(other: Tensor)

    /**
     * Subtract each element in [other] from the corresponding element in this tensor, in-place.
     *
     * @param other Tensor to subtract
     *
     * @throws IncompatibleShapeException if the tensors do not have the same [shape]
     */
    operator fun minusAssign(other: Tensor)

    /**
     * Multiplies all values in this tensor by [scalar], in-place.
     *
     * @param scalar Scalar value to multiply by
     */
    operator fun timesAssign(scalar: Float)

    /**
     * Divides all values in this tensor by [scalar], in-place.
     *
     * @param scalar Scalar value to divide by
     */
    operator fun divAssign(scalar: Float)

    /**
     * @param [function] Function to apply
     *
     * @return A new tensor with `function` applied elementwise to this tensor
     */
    fun map(function: (Float) -> Float): Tensor

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
    fun zip(other: Tensor, combine: (left: Float, right: Float) -> Float): Tensor

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
    fun zipAssign(other: Tensor, combine: (left: Float, right: Float) -> Float)

    /**
     * Calls [operation] with each element in the tensor, in row-major order (recursively from highest to lowest rank).
     * For example, given the following 3D tensor:
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
    fun reduce(operation: (accumulator: Float, element: Float) -> Float): Float

    /**
     * @return The sum over all elements in this tensor.
     */
    fun sum(): Float

    /**
     * @return The minimum (smallest) element in this tensor.
     */
    fun min(): Float

    /**
     * @return The maximum (largest) element in this tensor.
     */
    fun max(): Float

    /**
     * Sets all elements in the tensor to zero (`0f`)
     */
    fun zeroAssign()

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
    infix fun hadamard(other: Tensor): Tensor

    /**
     * Flattens the tensor down to a [Vector], in row-major order (recursively from highest to lowest rank). For
     * example, given the following 3D tensor:
     *
     * ```text
     * [ [1  2]   [5  6]
     *   [3  4]   [7  8] ]
     * ```
     *
     * It would flatten to (1, 2, 3, 4, 5, 6, 7, 8)
     *
     * @return Vector containing all tensor elements
     */
    fun flatten(): Vector
}

/**
 * [Tensor.times] with the arguments flipped. These operation are symmetric and equivalent.
 *
 * @param other Tensor to multiply by `this`
 *
 * @return New tensor `T` where `T[i]` = `other[i] * this`
 */
operator fun Float.times(other: Tensor) =
    other.map { it * this }

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

/**
 * Thrown by [BoundedTensor.castUnsafe], and methods which call it, when attempting to convert an incompatible generic
 * [Tensor] to that specific [BoundedTensor] type.
 *
 * Only tensors with equal [Tensor.shape] are compatible.
 */
open class IncompatibleGenericTensorException(message: String) : TensorOperationException(message)
