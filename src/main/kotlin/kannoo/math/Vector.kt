package kannoo.math

/**
 * 1-dimensional [Tensor] of elements (scalars). This is the lowest supported rank of [Tensor] and so [slices] is
 * inaccessible.
 */
@JvmInline
value class Vector(val elements: FloatArray) : Tensor<Vector> {

    /**
     * Tensor rank, always equal to `1` for vectors.
     */
    override val rank get() = 1

    /**
     * Number of elements in this vector.
     */
    override val size: Int get() = elements.size

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
     * @param tensor The tensor to sum with, must be a [Vector] of equal size
     *
     * @return A new vector `V` where `V[i]` = `this[i] + tensor[i]`
     *
     * @throws UnsupportedTensorOperation if [tensor] is not a [Vector] of equal size
     */
    override operator fun plus(tensor: Vector): Vector {
        return zip(tensor) { x, y -> x + y }
    }

    /**
     * @param tensor The tensor to subtract, must be a [Vector] of equal size
     *
     * @return A new vector `V` where `V[i]` = `this[i] - tensor[i]`
     *
     * @throws UnsupportedTensorOperation if [tensor] is not a [Vector] of equal size
     */
    override operator fun minus(tensor: Vector): Vector {
        return zip(tensor) { x, y -> x - y }
    }

    /**
     * @param scalar Scalar value to multiple by
     *
     * @return A new vector V where `V[i]` = `this[i] * scalar`
     */
    override operator fun times(scalar: Float): Vector =
        transform { it * scalar }

    /**
     * @param scalar Scalar value to divide by
     *
     * @return A new vector V where `V[i]` = `this[i] / scalar`
     */
    override operator fun div(scalar: Float): Vector =
        transform { it / scalar }

    /**
     * Add each element in [tensor] to the corresponding element in this vector, in-place.
     *
     * @param tensor The tensor to sum with, must be a [Vector] of equal size
     *
     * @throws UnsupportedTensorOperation if [tensor] is not a [Vector] of equal size
     */
    override operator fun plusAssign(tensor: Vector) {
        zipAssign(tensor) { x, y -> x + y }
    }

    /**
     * Subtract each element in [tensor] from the corresponding element in this vector, in-place.
     *
     * @param tensor The tensor to subtract, must be a [Vector] of equal size
     *
     * @throws UnsupportedTensorOperation if [tensor] is not a [Vector] of equal size
     */
    override operator fun minusAssign(tensor: Vector) {
        zipAssign(tensor) { x, y -> x - y }
    }

    /**
     * Multiplies all values in this vector by [scalar], in place.
     *
     * @param scalar Scalar value to multiple by
     */
    override operator fun timesAssign(scalar: Float) {
        assign { it * scalar }
    }

    /**
     * Divides all values in this vector by [scalar], in place.
     *
     * @param scalar Scalar value to multiple by
     */
    override operator fun divAssign(scalar: Float) {
        assign { it / scalar }
    }

    /**
     * @return A deep copy of this vector
     */
    override fun copy(): Vector =
        Vector(elements.copyOf())

    /**
     * @param [function] Function to apply
     *
     * @return A copy of this vector with [function] applied to each element
     */
    override fun transform(function: (Float) -> Float): Vector =
        Vector(size) { i -> function(this[i]) }

    /**
     * Applies [function] to all elements in this vector, in place.
     *
     * @param [function] Function to apply
     */
    override fun assign(function: (Float) -> Float) {
        for (i in 0 until size)
            this[i] = function(this[i])
    }

    // TODO: doc
    // TODO: generalize to tensor
    operator fun times(m: Matrix): Vector {
        if (size != m.rows)
            throw IllegalArgumentException("Vector size must equal matrix row count")

        val res = Vector(m.cols) { 0f }
        for (i in 0 until m.rows)
            for (j in 0 until m.cols)
                res[j] += m[i][j] * this[i]

        return res
    }

    // TODO: doc
    fun sum(): Float =
        elements.sum()

    // TODO: doc
    fun min(): Float =
        elements.min()

    // TODO: doc
    fun max(): Float =
        elements.max()

    // TODO: doc
    inline fun zip(other: Vector, crossinline combine: (left: Float, right: Float) -> Float): Vector {
        if (other.size != this.size)
            throw UnsupportedTensorOperation("Cannot combine vectors of different sizes")

        return Vector(size) { i ->
            combine(this[i], other[i])
        }
    }

    // TODO: doc
    inline fun zipAssign(other: Vector, crossinline combine: (left: Float, right: Float) -> Float) {
        if (other.size != this.size)
            throw UnsupportedTensorOperation("Cannot combine vectors of different sizes")

        for (i in 0 until size)
            this[i] = combine(this[i], other[i])
    }

    // TODO: doc
    inline fun zipSumOf(other: Vector, crossinline function: (Float, Float) -> Float): Float {
        if (other.size != this.size)
            throw UnsupportedTensorOperation("Cannot combine vectors of different sizes")

        var accumulator = 0f
        for (i in 0 until size)
            accumulator += function(this[i], other[i])

        return accumulator
    }
}

/**
 * @param size The size of the new vector
 *
 * @param init Initialization callback
 *
 * @return A new [Vector] of size [size] initialized by [init]
 */
inline fun Vector(size: Int, crossinline init: (index: Int) -> Float): Vector =
    Vector(FloatArray(size) { i -> init(i) })

/**
 * @param elements Elements of the new vector
 *
 * @return A new [Vector] containing the given elements
 */
fun vector(vararg elements: Float): Vector =
    Vector(elements)

// TODO: doc
fun tensor(vararg elements: Float): Vector =
    Vector(elements)

// TODO: doc
// TODO: generalize to tensor
fun hadamard(a: Vector, b: Vector): Vector = // TODO: verify same size
    Vector(a.size) { i -> a[i] * b[i] }

// TODO: doc
fun outer(a: Vector, b: Vector): Matrix =
    Matrix(rows = a.size, cols = b.size) { i, j -> a[i] * b[j] }
