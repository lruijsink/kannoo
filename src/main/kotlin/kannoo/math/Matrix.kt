package kannoo.math

/**
 * 2-dimensional [Tensor] composed of of [Vector] slices. By convention, the slices are stored row-first, meaning the
 * matrix element `M[i][j]` refers to the element in row `i` and column `j`.
 */
@JvmInline
value class Matrix(val rowVectors: Array<Vector>) : Composite<Matrix, Vector> {
    /**
     * Matrices are tensors of rank 2.
     */
    override val rank get() = 2

    /**
     * Equivalent to [rowVectors] but cast to `Array<Tensor>`. It is safe to read from this array, but only [Vector]
     * (and no other [Tensor] subtype) elements may be written to it.
     */
    @Suppress("UNCHECKED_CAST")
    override val slices get() = rowVectors.toList()

    override val size: Int get() = rowVectors.size

    val rows: Int get() = rowVectors.size

    val cols: Int get() = rowVectors[0].size

    override operator fun get(index: Int): Vector =
        rowVectors[index]

    override operator fun set(index: Int, slice: Vector) {
        rowVectors[index] = slice
    }

    override operator fun plus(tensor: Matrix): Matrix =
        if (tensor.rows != this.rows || tensor.cols != this.cols) throw IllegalArgumentException("Incompatible")
        else Matrix(rows) { i -> rowVectors[i] + tensor.rowVectors[i] }

    override operator fun minus(tensor: Matrix): Matrix =
        if (tensor.rows != this.rows || tensor.cols != this.cols) throw IllegalArgumentException("Incompatible")
        else Matrix(rows) { i -> rowVectors[i] - tensor.rowVectors[i] }

    override operator fun times(scalar: Float): Matrix =
        transform { it * scalar }

    override operator fun div(scalar: Float): Matrix =
        transform { it / scalar }

    override operator fun plusAssign(tensor: Matrix) {
        if (tensor.rows != this.rows || tensor.cols != this.cols) throw IllegalArgumentException("Incompatible")
        for (i in 0 until size) this[i].plusAssign(tensor[i])
    }

    override operator fun minusAssign(tensor: Matrix) {
        if (tensor.rows != this.rows || tensor.cols != this.cols) throw IllegalArgumentException("Incompatible")
        for (i in 0 until size) this[i].minusAssign(tensor[i])
    }

    override operator fun timesAssign(scalar: Float) {
        assign { it * scalar }
    }

    override operator fun divAssign(scalar: Float) {
        assign { it / scalar }
    }

    override fun transform(function: (Float) -> Float): Matrix =
        Matrix(rows) { i -> this[i].transform(function) }

    override fun assign(function: (Float) -> Float) {
        for (i in 0 until size) this[i].assign(function)
    }

    override fun copy(): Matrix =
        Matrix(rows) { this[it].copy() }

    // TODO: doc
    // TODO: generalize to tensor
    operator fun times(v: Vector): Vector {
        if (v.size != cols) throw UnsupportedTensorOperation("Vector size must equal matrix column count")
        val res = Vector(rows) { 0f }
        for (i in 0 until rows)
            for (j in 0 until cols)
                res[i] += this[i][j] * v[j]
        return res
    }

    // TODO: doc
    // TODO: add to Vector too
    inline fun forEachIndexed(crossinline action: (row: Int, col: Int) -> Unit) {
        for (i in 0 until rows)
            for (j in 0 until cols)
                action(i, j)
    }
}

inline fun Matrix(rows: Int, crossinline init: (row: Int) -> Vector): Matrix =
    Matrix(Array(rows) { row -> init(row) })

inline fun Matrix(rows: Int, cols: Int, crossinline init: () -> Float): Matrix =
    Matrix(Array(rows) { Vector(cols) { init() } })

inline fun Matrix(rows: Int, cols: Int, crossinline init: (row: Int, col: Int) -> Float): Matrix =
    Matrix(Array(rows) { row -> Vector(cols) { col -> init(row, col) } })

@Suppress("FINAL_UPPER_BOUND") // varargs of value classes are not normally allowed, hacky workaround
fun <T : Vector> matrix(vararg rowVectors: T): Matrix {
    @Suppress("UNCHECKED_CAST", "KotlinConstantConditions")
    return Matrix(rowVectors as Array<Vector>)
}

@Suppress("FINAL_UPPER_BOUND") // varargs of value classes are not normally allowed, hacky workaround
fun <T : Vector> tensor(vararg rowVectors: T): Matrix {
    @Suppress("UNCHECKED_CAST", "KotlinConstantConditions")
    return Matrix(rowVectors as Array<Vector>)
}

// TODO: doc
fun emptyMatrix(): Matrix =
    Matrix(arrayOf())
