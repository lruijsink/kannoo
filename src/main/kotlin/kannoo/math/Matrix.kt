package kannoo.math

/**
 * 2-dimensional [Tensor] composed of of [Vector] slices. By convention, the slices are stored row-first, meaning the
 * matrix element `M[i][j]` refers to the element in row `i` and column `j`.
 */
class Matrix(override val slices: Array<Vector>) : Composite<Matrix, Vector> {

    /**
     * Matrices are tensors of rank 2.
     */
    override val rank get() = 2

    override val shape get() = listOf(rows, cols)

    val rows: Int get() = slices.size

    val cols: Int get() = slices[0].size

    val rowVectors get() = slices

    override fun times(scalar: Float): Matrix =
        Matrix(rows, cols) { i, j -> this[i][j] * scalar }

    override fun plusAssign(tensor: Matrix) {
        for (i in 0 until rows)
            for (j in 0 until cols)
                this[i][j] += tensor[i][j]
    }

    override fun minusAssign(tensor: Matrix) {
        for (i in 0 until rows)
            for (j in 0 until cols)
                this[i][j] -= tensor[i][j]
    }

    override fun map(function: (Float) -> Float): Matrix =
        Matrix(rows) { i -> this[i].map(function) }

    override fun mapAssign(function: (Float) -> Float) {
        for (i in 0 until size) this[i].mapAssign(function)
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

    override fun zip(other: Matrix, combine: (Float, Float) -> Float): Matrix =
        Matrix(rows) { i -> this[i].zip(other[i], combine) }

    override fun zipAssign(other: Matrix, combine: (Float, Float) -> Float) {
        for (i in 0 until rows)
            this[i].zipAssign(other[i], combine)
    }
}

inline fun Matrix(rows: Int, crossinline init: (row: Int) -> Vector): Matrix =
    Matrix(Array(rows) { row -> init(row) })

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
