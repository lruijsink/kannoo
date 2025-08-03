package kannoo

typealias Matrix = List<DoubleArray>

fun Matrix(w: Int, h: Int, init: (Int, Int) -> Double): Matrix =
    List(w) { i -> Vector(h) { j -> init(i, j) } }

fun Matrix(w: Int, h: Int, init: () -> Double): Matrix =
    Matrix(w, h) { _, _ -> init() }

val Matrix.rows get() = size
val Matrix.cols get() = if (rows == 0) 0 else this[0].size

fun Matrix.forEachIndexedCell(fn: (i: Int, j: Int) -> Unit) {
    for (i in 0 until rows)
        for (j in 0 until cols)
            fn(i, j)
}

fun emptyMatrix(): Matrix =
    listOf()

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

fun Matrix.addInPlace(m: Matrix) {
    if (rows != m.rows || cols != m.cols) throw IllegalArgumentException("Matrices must have same dimensions")
    forEachIndexedCell { i, j -> this[i][j] += m[i][j] }
}

fun Matrix.subInPlace(m: Matrix) {
    if (rows != m.rows || cols != m.cols) throw IllegalArgumentException("Matrices must have same dimensions")
    forEachIndexedCell { i, j -> this[i][j] -= m[i][j] }
}

fun Matrix.mulInPlace(s: Double) {
    forEachIndexedCell { i, j -> this[i][j] *= s }
}
