package kannoo.math2

/**
 * 2-dimensional [Tensor] composed of of [Vector] slices. By convention, the slices are stored row-first, meaning the
 * matrix element `M[i][j]` refers to the element in row `i` and column `j`.
 */
@JvmInline
value class Matrix(val rowVectors: Array<Vector>) : Tensor {
    /**
     * Matrices are tensors of rank 2.
     */
    override val rank get() = 2

    /**
     * Equivalent to [rowVectors] but cast to `Array<Tensor>`. It is safe to read from this array, but only [Vector]
     * (and no other [Tensor] subtype) elements may be written to it.
     */
    @Suppress("UNCHECKED_CAST")
    override val slices get() = rowVectors as Array<Tensor>

    override val size: Int get() = rowVectors.size

    val rows: Int get() = rowVectors.size

    val cols: Int get() = rowVectors[0].size

    operator fun get(index: Int): Vector =
        rowVectors[index]

    operator fun set(index: Int, vector: Vector) {
        rowVectors[index] = vector
    }

    override operator fun plus(t: Tensor): Matrix =
        if (t !is Matrix || t.rows != this.rows || t.cols != this.cols) throw IllegalArgumentException("Incompatible")
        else Matrix(rows) { i -> rowVectors[i] + t.rowVectors[i] }

    override operator fun minus(t: Tensor): Matrix =
        if (t !is Matrix || t.rows != this.rows || t.cols != this.cols) throw IllegalArgumentException("Incompatible")
        else Matrix(rows) { i -> rowVectors[i] - t.rowVectors[i] }

    override operator fun times(s: Float): Matrix =
        map { it * s }

    override operator fun div(s: Float): Matrix =
        map { it / s }

    override operator fun plusAssign(t: Tensor) {
        if (t !is Matrix || t.rows != this.rows || t.cols != this.cols) throw IllegalArgumentException("Incompatible")
        for (i in 0 until size) this[i].plusAssign(t[i])
    }

    override operator fun minusAssign(t: Tensor) {
        if (t !is Matrix || t.rows != this.rows || t.cols != this.cols) throw IllegalArgumentException("Incompatible")
        for (i in 0 until size) this[i].minusAssign(t[i])
    }

    override operator fun timesAssign(s: Float) {
        mapAssign { it * s }
    }

    override operator fun divAssign(s: Float) {
        mapAssign { it / s }
    }

    override fun map(transform: (Float) -> Float): Matrix =
        Matrix(rows) { i -> this[i].map(transform) }

    override fun mapAssign(transform: (Float) -> Float) {
        for (i in 0 until size) this[i].mapAssign(transform)
    }

    override fun copy(): Tensor =
        Matrix(rows) { this[it].copy() }
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
