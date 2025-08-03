package kannoo

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
