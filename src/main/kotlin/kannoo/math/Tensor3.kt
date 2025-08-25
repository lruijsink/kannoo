@file:Suppress("FunctionName")

package kannoo.math

/**
 * Rank 3 tensor; an [NTensor] with [Matrix] slices.
 */
typealias Tensor3 = NTensor<Matrix>

// TODO: doc
inline fun NTensor(s0: Int, s1: Int, s2: Int, crossinline initialize: (Int, Int, Int) -> Float): Tensor3 =
    NTensor(s0) { h -> Matrix(s1, s2) { i, j -> initialize(h, i, j) } }

// TODO: doc
inline fun Tensor3(s0: Int, s1: Int, s2: Int, crossinline initialize: (Int, Int, Int) -> Float): Tensor3 =
    NTensor(s0, s1, s2, initialize)

// TODO: doc
inline fun Tensor3(size: Int, crossinline initialize: (i: Int) -> Matrix): Tensor3 =
    NTensor(size, initialize)

// TODO: doc
fun NTensor(s0: Int, s1: Int, s2: Int): Tensor3 =
    NTensor(s0) { Matrix(s1, s2) }

// TODO: doc
fun Tensor3(s0: Int, s1: Int, s2: Int): Tensor3 =
    Tensor3(s0) { Matrix(s1, s2) }

// TODO: doc
fun randomTensor(s0: Int, s1: Int, s2: Int): Tensor3 =
    NTensor(s0) { randomMatrix(s1, s2) }
