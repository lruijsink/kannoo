package kannoo.math

/**
 * 1-dimensional [Tensor] of elements (scalars). This is the lowest supported rank of [Tensor].
 *
 * @param elements Element array to initialize the vector with. This array is part of the public API of this class and
 * may be written to.
 */
class Vector(val elements: FloatArray) : BoundedTensor<Vector> {

    /**
     * Tensor rank, always equal to `1` for vectors.
     */
    override val rank: Int get() = 1

    /**
     * Number of elements in this vector.
     */
    override val size: Int get() = elements.size

    /**
     * Tensor shape, for vectors this is just a single dimension: its [size].
     */
    override val shape: Shape get() = Shape(size)

    /**
     * @return A deep copy of this vector
     */
    override fun copy(): Vector =
        Vector(elements.copyOf())

    /**
     * @param index The index of the element to get
     *
     * @return The element at index [index]
     */
    operator fun get(index: Int): Float =
        elements[index]

    /**
     * Set the element at index [index] to [value]
     *
     * @param index The index to modify
     *
     * @param value The value to set the element to
     */
    operator fun set(index: Int, value: Float) {
        elements[index] = value
    }

    /**
     * @param other Vector to sum with
     *
     * @return New vector `V` where `V[i]` = `this[i] + other[i]`
     *
     * @throws IncompatibleShapeException if the vectors do not have the same [size]
     */
    override fun plus(other: Vector): Vector {
        if (other.size != this.size)
            throw IncompatibleShapeException("Cannot combine vectors of different sizes")

        return Vector(size) { i -> this[i] + other[i] }
    }

    /**
     * @param other Vector to subtract
     *
     * @return New vector `V` where `V[i]` = `this[i] - other[i]`
     *
     * @throws IncompatibleShapeException if the vectors do not have the same [size]
     */
    override fun minus(other: Vector): Vector {
        if (other.size != this.size)
            throw IncompatibleShapeException("Cannot combine vectors of different sizes")

        return Vector(size) { i -> this[i] - other[i] }
    }

    /**
     * @param scalar Scalar value to multiply by
     *
     * @return New vector `V` where `V[i]` = `this[i] * scalar`
     */
    override fun times(scalar: Float): Vector =
        Vector(size) { i -> this[i] * scalar }

    /**
     * @param scalar Scalar value to divide by
     *
     * @return New vector `V` where `V[i]` = `this[i] / scalar`
     */
    override fun div(scalar: Float): Vector =
        Vector(size) { i -> this[i] / scalar }

    /**
     * Add each element in [other] to the corresponding element in this vector, in-place.
     *
     * @param other Vector to sum with
     *
     * @throws IncompatibleShapeException if the vectors do not have the same [size]
     */
    override fun plusAssign(other: Vector) {
        if (other.size != this.size)
            throw IncompatibleShapeException("Cannot combine vectors of different sizes")

        for (i in 0 until size)
            this[i] += other[i]
    }

    /**
     * Subtract each element in [other] from the corresponding element in this vector, in-place.
     *
     * @param other Vector to subtract
     *
     * @throws IncompatibleShapeException if the vectors do not have the same [size]
     */
    override fun minusAssign(other: Vector) {
        if (other.size != this.size)
            throw IncompatibleShapeException("Cannot combine vectors of different sizes")

        for (i in 0 until size)
            this[i] -= other[i]
    }

    /**
     * Multiplies all values in this vector by [scalar], in-place.
     *
     * @param scalar Scalar value to multiple by
     */
    override fun timesAssign(scalar: Float) {
        for (i in 0 until size)
            this[i] *= scalar
    }

    /**
     * Divides all values in this vector by [scalar], in-place.
     *
     * @param scalar Scalar value to divide by
     */
    override fun divAssign(scalar: Float) {
        for (i in 0 until size)
            this[i] /= scalar
    }

    /**
     * @param [function] Function to apply
     *
     * @return A copy of this vector with [function] applied to each element
     */
    override fun map(function: (Float) -> Float): Vector =
        Vector(size) { i -> function(this[i]) }

    /**
     * Applies [function] to all elements in this vector, in place.
     *
     * @param [function] Function to apply
     */
    override fun mapAssign(function: (Float) -> Float) {
        for (i in 0 until size)
            this[i] = function(this[i])
    }

    /**
     * Zips this vector with another and applies the combining operation over each (element, element) pair in both
     * vectors.
     *
     * @param other Vector to zip with
     *
     * @param combine Operation to apply to each zipped (element, element) pair
     *
     * @return New vector `V` where for each element `i`:
     *
     * `V[i]` = `combine(this[i], other[i])`
     *
     * @throws IncompatibleShapeException if the vectors do not have the same [size]
     */
    override fun zip(other: Vector, combine: (left: Float, right: Float) -> Float): Vector {
        if (other.size != this.size)
            throw IncompatibleShapeException("Cannot combine vectors of different sizes")

        return Vector(size) { i ->
            combine(this[i], other[i])
        }
    }

    /**
     * Zips this vector with another and applies the combining operation over each (element, element) pair in-place,
     * such that for each element `i`:
     *
     * `this[i]` = `combine(this[i], other[i])`
     *
     * @param other Vector to zip with
     *
     * @param combine Operation to apply to each zipped (element, element) pair
     *
     * @throws IncompatibleShapeException if the vectors do not have the same [size]
     */
    override fun zipAssign(other: Vector, combine: (left: Float, right: Float) -> Float) {
        if (other.size != this.size)
            throw IncompatibleShapeException("Cannot combine vectors of different sizes")

        for (i in 0 until size)
            this[i] = combine(this[i], other[i])
    }

    /**
     * Calls [operation] with each element in the vector, in order.
     *
     * @param operation Function to call
     */
    override fun forEachElement(operation: (element: Float) -> Unit) {
        elements.forEach(operation)
    }

    /**
     * Vectors are already flat so this produces the vector as-is, but note that this produces a copy rather than
     * returning the same object.
     *
     * @return Copy of this vector
     */
    override fun flatten(): Vector =
        copy()

    /**
     * Calls [operation] with each element in the vector, and its respective index, in order.
     *
     * @param operation Function to call
     */
    inline fun forEachIndexed(crossinline operation: (index: Int, element: Float) -> Unit) {
        elements.forEachIndexed(operation)
    }

    /**
     * Computes the sum of applying [combine] to every (element, element) pair from this vector and [other].
     *
     * Equivalent to [zip]`(other, function).sum()` but without allocating the intermediate vector that [zip] would.
     *
     * @param other Vector to zip with
     *
     * @param combine Operation to apply to each zipped (element, element) pair
     *
     * @return sum over `compute(this[i], other[i])` for all `i`
     *
     * @throws IncompatibleShapeException if the vectors do not have the same [size]
     */
    inline fun zipSumOf(other: Vector, crossinline combine: (Float, Float) -> Float): Float {
        if (other.size != this.size)
            throw IncompatibleShapeException("Cannot combine vectors of different sizes")

        var accumulator = 0f
        for (i in 0 until size)
            accumulator += combine(this[i], other[i])

        return accumulator
    }

    // TODO: generalize this to any Tensor * Tensor
    /**
     * Computes the vector-matrix multiplication of this vector and [matrix]. Note that this is equivalent to
     * [matrix]`.transpose() * this`
     *
     * @param matrix The matrix to multiply with
     *
     * @return New vector `V` where for each element (i = row, j = column) in [matrix]:
     *
     * `V[j]` = `this[i] * matrix[i, j]`
     *
     * @throws IncompatibleShapeException if the matrix does not have row count [size]
     */
    operator fun times(matrix: Matrix): Vector {
        if (size != matrix.rows)
            throw IncompatibleShapeException("Vector size must equal matrix row count")

        val res = Vector(matrix.cols)
        for (i in 0 until matrix.rows)
            for (j in 0 until matrix.cols)
                res[j] += matrix[i, j] * this[i]

        return res
    }

    /**
     * Computes the outer product of this vector and [other].
     *
     * @param other Vector to take the cross-product of
     *
     * @return Matrix `M` of `this.`[size] rows and `other.size` columns, defined as:
     *
     * `M[i, j] = this[i] * other[j]`
     */
    infix fun outer(other: Vector): Matrix =
        Matrix(rows = this.size, cols = other.size) { i, j -> this[i] * other[j] }

    // TODO: doc
    fun unFlatten(shape: Shape, offset: Int = 0): Tensor {
        if (totalElements > this.size)
            throw IllegalArgumentException("Insufficient number of elements for shape $this")

        return when (shape.rank) {
            1 ->
                Vector(elements.copyOfRange(offset, shape[0]))

            2 -> {
                val (rows, cols) = shape
                Matrix(rows, cols) { i, j -> this[i * cols + j + offset] }
            }

            3 -> {
                val (size, rows, cols) = shape
                val elementsPerMatrix = rows * cols
                NTensor(size) { n ->
                    Matrix(rows, cols) { i, j ->
                        this[n * elementsPerMatrix + i * cols + j + offset]
                    }
                }
            }

            else -> {
                val elementsPerSlice = totalElements / shape[0]
                NTensor(size = shape[0]) { n ->
                    unFlatten(shape.sliceShape, offset = n * elementsPerSlice) as NTensor<*>
                }
            }
        }
    }

    override fun toString(): String =
        elements.toList().toString()

    override fun equals(other: Any?): Boolean =
        other is Vector && elements.contentEquals(other.elements)

    override fun hashCode(): Int =
        elements.contentHashCode()

    fun prettyPrint(useCommas: Boolean = false): String {
        val els = elements
            .map { it.toString() }
            .map { if (it.endsWith(".0")) it.dropLast(2) else it }

        val w = els.maxOf { it.length }
        val p = els.map { it.padStart(w, ' ') }
        return "[ ${p.joinToString(if (useCommas) ", " else "  ")} ]"
    }

    fun prettyPrint(cellWidth: Int, useCommas: Boolean = false): String {
        val p = elements
            .map { it.toString() }
            .map { if (it.endsWith(".0")) it.dropLast(2) else it }
            .map { if (it.length > cellWidth) it.substring(0, cellWidth) else it.padStart(cellWidth, ' ') }

        return "[ ${p.joinToString(if (useCommas) ", " else "  ")} ]"
    }
}

/**
 * @param size Size of the new vector
 *
 * @param init Initialization callback
 *
 * @return New Vector V defined as:
 *
 * `V[i] = `[init]`(i)`
 */
inline fun Vector(size: Int, crossinline init: (index: Int) -> Float): Vector =
    Vector(FloatArray(size) { i -> init(i) })

/**
 * @param size Size of the new vector
 *
 * @return New Vector V of size [size] with all elements set to `0.0f`
 */
fun Vector(size: Int): Vector =
    Vector(FloatArray(size))

/**
 * @param elements Elements of the new vector
 *
 * @return New [Vector] containing all [elements] in the given order
 */
fun vector(vararg elements: Float): Vector =
    Vector(elements)

/**
 * [Vector] overload of [tensor]`(...)`
 *
 * @param elements Elements of the new vector (tensor)
 *
 * @return New [Vector] containing all [elements] in the given order
 */
fun tensor(vararg elements: Float): Vector =
    Vector(elements)

/**
 * @return Empty vector
 *
 * WARNING: Some operations do not support empty vectors
 */
fun emptyVector(): Vector =
    Vector(floatArrayOf())
