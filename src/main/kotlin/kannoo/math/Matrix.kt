package kannoo.math

@JvmInline
value class Matrix(private val arr: Array<Vector>) {

    constructor(rows: Int, cols: Int, init: (row: Int, col: Int) -> Double)
            : this(Array(rows) { row -> Vector(cols) { col -> init(row, col) } })

    constructor(rows: Int, cols: Int, init: () -> Double)
            : this(rows, cols, { _, _ -> init() })

    val rows
        get(): Int = arr.size

    val cols
        get(): Int = if (rows == 0) 0 else arr[0].size

    val rowVectors
        get(): Array<Vector> = arr

    operator fun get(index: Int): Vector =
        arr[index]

    operator fun times(rhs: Vector): Vector {
        if (rhs.size != cols) throw IllegalArgumentException("Vector size must equal column count")
        val res = Vector(rows)
        for (i in 0 until rows)
            for (j in 0 until cols)
                res[i] += this[i][j] * rhs[j]
        return res
    }

    operator fun times(scalar: Double): Matrix =
        Matrix(rows, cols) { row, col -> this[row][col] * scalar }

    operator fun plusAssign(rhs: Matrix) {
        if (rows != rhs.rows || cols != rhs.cols) throw IllegalArgumentException("Matrices must have same dimensions")
        forEachIndexed { row, col -> this[row][col] += rhs[row][col] }
    }

    operator fun minusAssign(rhs: Matrix) {
        if (rows != rhs.rows || cols != rhs.cols) throw IllegalArgumentException("Matrices must have same dimensions")
        forEachIndexed { row, col -> this[row][col] -= rhs[row][col] }
    }

    operator fun timesAssign(rhs: Double) {
        forEachIndexed { row, col -> this[row][col] *= rhs }
    }

    fun forEachIndexed(fn: (row: Int, col: Int) -> Unit) {
        for (row in 0 until rows)
            for (col in 0 until cols)
                fn(row, col)
    }

    fun zero() {
        forEachIndexed { row, col -> this[row][col] = 0.0 }
    }

    fun transpose(): Matrix =
        Matrix(cols, rows) { i, j -> this[j][i] }
}

fun emptyMatrix(): Matrix =
    Matrix(arrayOf())

fun randomMatrix(w: Int, h: Int): Matrix =
    Matrix(w, h) { randomDouble() }

fun outer(a: Vector, b: Vector): Matrix =
    Matrix(a.size, b.size) { i, j -> a[i] * b[j] }

fun transposeDot(m: Matrix, v: Vector): Vector {
    if (v.size != m.rows) throw IllegalArgumentException("nn.Matrix row count must equal vector size")
    return Vector(m.cols) { j ->
        (0 until m.rows).sumOf { i -> m[i][j] * v[i] }
    }
}

inline fun <T> Iterable<T>.sumOfMatrix(selector: (T) -> Matrix): Matrix {
    val first = this.firstOrNull() ?: return emptyMatrix()
    val sum = selector(first)
    this.forEachIndexed { i, el ->
        if (i > 0) sum += selector(el)
    }
    return sum
}

fun Iterable<Matrix>.sum(): Matrix {
    val sum = this.firstOrNull() ?: return emptyMatrix()
    this.forEachIndexed { i, matrix ->
        if (i > 0) sum += matrix
    }
    return sum
}

operator fun Double.times(matrix: Matrix): Matrix = matrix * this

operator fun Vector.times(m: Matrix): Vector {
    if (size != m.rows) throw IllegalArgumentException("Vector size must equal row count")
    val res = Vector(m.cols)
    for (i in 0 until m.rows)
        for (j in 0 until m.cols)
            res[j] += m[i][j] * this[i]
    return res
}
