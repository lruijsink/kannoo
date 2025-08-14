package kannoo.math

/**
 * 2-dimensional [Tensor] composed of of [Vector] row vector slices. By convention, the slices are stored row-first,
 * meaning the matrix element `M[i, j]` refers to the element in row `i` and column `j`.
 */
class Matrix(override val slices: Array<Vector>) : Composite<Matrix, Vector> {

    init {
        if (slices.any { it.size != slices[0].size })
            throw IncompatibleShapeException("All row vectors in a matrix must have the same size")
    }

    /**
     * Matrices are tensors of rank 2.
     */
    override val rank: Int get() = 2

    /**
     * Size of the tensor, which is its number of [slices] ([rowVectors] for a matrix).
     */
    override val size: Int get() = slices.size

    /**
     * The tensor shape, equal to ([rows], [cols]) for matrices.
     *
     * The matrix has [rows] slices (row vectors) which are each [cols] elements wide.
     */
    override val shape get(): List<Int> = listOf(rows, cols)

    /**
     * Number of rows in this matrix. Equivalent to the number of [slices], or [rowVectors].
     */
    val rows: Int get() = slices.size

    /**
     * Number of columns in this matrix. Equivalent to the [size] of each row vector.
     */
    val cols: Int get() = slices[0].size

    /**
     * Row vectors of this matrix. Equivalent to its [slices].
     */
    val rowVectors: Array<Vector> get() = slices

    /**
     * @param index Row vector index to get
     *
     * @return Row vector at index [index]
     */
    override operator fun get(index: Int): Vector =
        slices[index]

    /**
     * @param index Row vector index to set
     *
     * @param slice New row vector value
     */
    override operator fun set(index: Int, slice: Vector) {
        slices[index] = slice
    }

    /**
     * Indexed read, equivalent to `this[i][j]`
     *
     * @param i Row index

     * @param j Column index

     * @return Element with index (i, j)
     */
    operator fun get(i: Int, j: Int): Float =
        slices[i][j]

    /**
     * Indexed write, equivalent to `this[i][j] = value`
     *
     * @param i Row index

     * @param j Column index
     */
    operator fun set(row: Int, col: Int, value: Float) {
        slices[row][col] = value
    }

    /**
     * @return A deep copy of this matrix and its component slices (vectors).
     */
    override fun copy(): Matrix =
        Matrix(Array(rows) { this[it].copy() })

    /**
     * @param other Matrix to sum with
     *
     * @return New matrix `M` where `M[i, j]` = `this[i, j] + other[i, j]`
     *
     * @throws IncompatibleShapeException if the matrices do not have the same [shape]
     */
    override operator fun plus(other: Matrix): Matrix =
        zip(other) { x, y -> x + y }

    /**
     * @param other Matrix to subtract
     *
     * @return New matrix `M` where `M[i, j]` = `this[i, j] - other[i, j]`
     *
     * @throws IncompatibleShapeException if the matrices do not have the same [shape]
     */
    override operator fun minus(other: Matrix): Matrix =
        zip(other) { x, y -> x - y }

    /**
     * @param scalar Scalar value to multiply by
     *
     * @return New matrix `M` where `M[i, j]` = `this[i, j] * scalar`
     */
    override operator fun times(scalar: Float): Matrix =
        map { x -> x * scalar }

    /**
     * @param scalar Scalar value to divide by
     *
     * @return New matrix `M` where `M[i, j]` = `this[i, j] / scalar`
     */
    override operator fun div(scalar: Float): Matrix =
        map { x -> x / scalar }

    /**
     * Add each element in [other] to the corresponding element in this matrix, in-place.
     *
     * @param other Matrix to sum with
     *
     * @throws IncompatibleShapeException if the matrices do not have the same [shape]
     */
    override operator fun plusAssign(other: Matrix) {
        zipAssign(other) { x, y -> x + y }
    }

    /**
     * Subtract each element in [other] from the corresponding element in this matrix, in-place.
     *
     * @param other Matrix to subtract
     *
     * @throws IncompatibleShapeException if the matrices do not have the same [shape]
     */
    override operator fun minusAssign(other: Matrix) {
        zipAssign(other) { x, y -> x - y }
    }

    /**
     * Multiplies all values in this matrix by [scalar], in-place.
     *
     * @param scalar Scalar value to multiply by
     */
    override operator fun timesAssign(scalar: Float) {
        mapAssign { it * scalar }
    }

    /**
     * Divides all values in this matrix by [scalar], in-place.
     *
     * @param scalar Scalar value to divide by
     */
    override operator fun divAssign(scalar: Float) {
        mapAssign { it / scalar }
    }

    /**
     * @param [function] Function to apply
     *
     * @return A new matrix with `function` applied elementwise to this matrix
     */
    override fun map(function: (Float) -> Float): Matrix =
        Matrix(rows, cols) { i, j ->
            function(this[i, j])
        }

    /**
     * Applies [function] to all scalar elements in this matrix, in-place.
     *
     * @param [function] Function to apply
     */
    override fun mapAssign(function: (Float) -> Float) {
        for (i in 0 until rows)
            for (j in 0 until cols)
                this[i, j] = function(this[i, j])
    }

    /**
     * Zips this matrix with another and applies the combining operation over each element pair in both.
     *
     * @param other Matrix to zip with
     *
     * @param combine Operation to apply to each zipped (i, j) pair
     *
     * @return New tensor `M` where for each pair (i, j):
     *
     * `T[i, j]` = `combine(this[i, j], other[i, j])`
     *
     * @throws IncompatibleShapeException if the matrices do not have the same [shape]
     */
    override fun zip(other: Matrix, combine: (Float, Float) -> Float): Matrix {
        if (this.rows != other.rows || this.cols != other.cols)
            throw IncompatibleShapeException("Cannot combine matrices with different dimensions")

        return Matrix(rows, cols) { i, j ->
            combine(this[i, j], other[i, j])
        }
    }

    /**
     * Zips this matrix with another and applies the combining operation to each element in this matrix.
     *
     * `this[i]` = `combine(this[i], other[i])`
     *
     * @param other Matrix to zip with
     *
     * @param combine Operation to apply to each zipped (element, element) pair
     *
     * @throws IncompatibleShapeException if the matrices do not have the same [shape]
     */
    override fun zipAssign(other: Matrix, combine: (Float, Float) -> Float) {
        if (this.rows != other.rows || this.cols != other.cols)
            throw IncompatibleShapeException("Cannot combine matrices with different dimensions")

        for (i in 0 until rows)
            for (j in 0 until cols)
                this[i, j] = combine(this[i, j], other[i, j])
    }

    /**
     * Calls [operation] with each element in the matrix, in row-major order. For example, given the matrix:
     *
     * ```text
     * [1  2  3]
     * [4  5  6]
     * ```
     *
     * The call order would be (1, 2, 3, 4, 5, 6)
     *
     * @param operation Function to call
     */
    override fun forEachElement(operation: (Float) -> Unit) {
        for (i in 0 until rows)
            for (j in 0 until cols)
                operation(this[i, j])
    }

    // TODO: generalize this to any Tensor * Tensor
    /**
     * Computes the matrix-vector multiplication of this matrix and [vector]. Note that this is equivalent to
     * [vector]` * this.transpose()`
     *
     * @param vector The vector to multiply with
     *
     * @return New vector `V` where for each element (i = row, j = column) in this matrix:
     *
     * `V[i]` = `this[i, j] * vector[j]`
     *
     * @throws IncompatibleShapeException if the vector does not have [cols] elements
     */
    operator fun times(vector: Vector): Vector {
        if (vector.size != cols)
            throw IncompatibleShapeException("Vector size must equal matrix column count")

        val res = Vector(rows)
        for (i in 0 until rows)
            for (j in 0 until cols)
                res[i] += this[i, j] * vector[j]
        return res
    }

    /**
     * Calls [operation] with each element in the matrix, and its respective index pair, in order.
     *
     * @param operation Function to call
     */
    inline fun forEachIndexed(crossinline operation: (row: Int, col: Int) -> Unit) {
        for (i in 0 until rows)
            for (j in 0 until cols)
                operation(i, j)
    }

    /**
     * @return New matrix `M` which is this one transposed, such that:
     *
     * `M[i, j] = this[j, i]`
     */
    fun transpose(): Matrix =
        Matrix(cols, rows) { i, j -> this[j, i] }
}

/**
 * @param rows Row count
 *
 * @param cols Column count
 *
 * @return New [rows] x [cols] matrix with all elements set to `0.0f`
 */
fun Matrix(rows: Int, cols: Int): Matrix =
    Matrix(Array(rows) { Vector(cols) })

/**
 * @param rows Row count
 *
 * @param cols Column count
 *
 * @param init Initialization function
 *
 * @return New [rows] x [cols] matrix with all elements initialized as [init]`(row, col)`
 */
inline fun Matrix(rows: Int, cols: Int, crossinline init: (row: Int, col: Int) -> Float): Matrix =
    Matrix(Array(rows) { row -> Vector(cols) { col -> init(row, col) } })

/**
 * @param vectors Row vectors to initialize with
 *
 * @return Matrix of [vectors] row vectors
 */
@Suppress("FINAL_UPPER_BOUND") // varargs of value classes are not normally allowed, hacky workaround
fun <T : Vector> matrix(vararg vectors: T): Matrix {
    @Suppress("UNCHECKED_CAST", "KotlinConstantConditions")
    return Matrix(vectors as Array<Vector>)
}

/**
 * [Matrix] overload of [tensor]`(...)`
 *
 * @param vectors Row vectors to initialize with
 *
 * @return Matrix of [vectors] row vectors
 */
@Suppress("FINAL_UPPER_BOUND") // varargs of value classes are not normally allowed, hacky workaround
fun <T : Vector> tensor(vararg rowVectors: T): Matrix {
    @Suppress("UNCHECKED_CAST", "KotlinConstantConditions")
    return Matrix(rowVectors as Array<Vector>)
}

/**
 * @return Empty matrix
 *
 * WARNING: Some operations do not support empty matrices
 */
fun emptyMatrix(): Matrix =
    Matrix(arrayOf())
