package kannoo.math

/**
 * N-dimensional tensor, composed of [slices] with [rank] N - 1, which must themselves be [Composite].
 *
 * Note that this makes [Matrix] the only other [Composite] tensor type, and the only one which accepts non-composite
 * slices (of type [Vector]), thereby effectively setting the minimum rank of [NTensor] to 3.
 *
 * @param T Slice tensor type, must be [Composite]
 *
 * @param S Nested slice tensor type; the slice type of [T]
 */
class NTensor<T>(override val slices: Array<T>) : Composite<NTensor<T>, T> where T : Composite<T, *> {

    init {
        if (slices.any { it.size != slices[0].size })
            throw IncompatibleShapeException("All slices in an NTensor must have the same size")
    }

    /**
     * The rank (dimensions) of this tensor, min. 3 for an [NTensor]
     */
    override val rank: Int
        get() = slices[0].rank + 1

    /**
     * The tensor's size in its highest dimension. For example: a 3D tensor of 5 matrices has size 5, regardless of the
     * dimensions of those matrices. Equivalent to [shape]`[0]`
     */
    override val size: Int
        get() = slices.size

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
    override val shape: Shape
        get() = Shape(compositeSize = size, sliceShape = slices[0].shape)

    /**
     * @return A deep copy of this tensor, with equal rank, dimensions, and element values
     */
    override fun copy(): NTensor<T> {
        val slicesCopy = slices.copyOf()
        for (i in 0 until size)
            slicesCopy[i] = slices[i].copy()
        return NTensor(slicesCopy)
    }

    /**
     * @param index Slice index to get
     *
     * @return Slice at index [index]
     */
    override operator fun get(index: Int): T =
        slices[index]

    /**
     * @param index Slice index to set
     *
     * @param slice New slice value
     */
    override operator fun set(index: Int, slice: T) {
        slices[index] = slice
    }

    /**
     * @param [function] Function to apply
     *
     * @return A new tensor with `function` applied elementwise to this tensor
     */
    override fun map(function: (Float) -> Float): NTensor<T> =
        mapIndexed { i -> this[i].map(function) }

    /**
     * Applies [function] to all scalar elements in this tensor, in-place.
     *
     * @param [function] Function to apply
     */
    override fun mapAssign(function: (Float) -> Float) {
        for (i in 0 until size) this[i].mapAssign(function)
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
    override fun zip(other: NTensor<T>, combine: (Float, Float) -> Float): NTensor<T> =
        mapIndexed { i -> this[i].zip(other[i], combine) }

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
    override fun zipAssign(other: NTensor<T>, combine: (Float, Float) -> Float) {
        for (i in 0 until size)
            this[i].zipAssign(other[i], combine)
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
    override fun forEachElement(operation: (Float) -> Unit) {
        slices.forEach { it.forEachElement(operation) }
    }

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
    override fun flatten(): Vector {
        val res = Vector(totalElements)
        var c = 0
        forEachElement { res[c++] = it }
        return res
    }

    /**
     * @param function Produces slice values
     *
     * @return New Tensor T of equal [shape] as this, and slices set to `T[i]` = [function]`(T[i])`
     */
    private inline fun mapIndexed(crossinline function: (Int) -> T): NTensor<T> {
        val res = copy() // Can't construct a new Array<NTensor<T>> due to type erasure, but we can copy the existing
        res.assignIndexed(function)
        return res
    }

    /**
     * Assigns new values to each slice, per slice index, such that `T[i]` = [function]`(T[i])`
     *
     * @param function Produces slice values
     */
    private inline fun assignIndexed(crossinline function: (Int) -> T) {
        for (i in 0 until size)
            this[i] = function(i)
    }

    override fun toString(): String =
        slices.toList().toString()

    override fun equals(other: Any?): Boolean =
        other is NTensor<T> && slices.contentEquals(other.slices)

    override fun hashCode(): Int =
        slices.contentHashCode()
}

/**
 * Constructs a new [NTensor] with [size] slices defined by [initialize].
 *
 * @param size Number of slices to instantiate
 *
 * @param initialize Function used to instantiate each slice
 *
 * @param T Slice tensor type
 *
 * @return Tensor of [size] slices, each instantiated by [initialize]
 *
 * @throws IncompatibleShapeException If [initialize] produces slices of differing shapes
 */
inline fun <reified T : Composite<T, *>> NTensor(size: Int, crossinline initialize: (index: Int) -> T): NTensor<T> =
    NTensor(Array(size) { i -> initialize(i) })

/**
 * Constructs a new [NTensor] composed of [slices]
 *
 * @param T Slice tensor type, must itself be [Composite]
 *
 * @param S Nested slice tensor type; the slice type of [T]
 *
 * @return Rank N + 1 tensor, where N = [T]'s rank, containing [slices]
 */
fun <T : Composite<T, S>, S : BoundedTensor<S>> tensor(vararg slices: T): NTensor<T> {
    @Suppress("KotlinConstantConditions") // We know this cast is safe:
    return NTensor(slices as Array<T>)
}
