@file:Suppress("FunctionName")

package kannoo.math

/**
 * Rank 4 tensor; an [NTensor] with [Tensor3] slices.
 */
typealias Tensor4 = NTensor<Tensor3>

// TODO: doc
inline fun NTensor(s0: Int, s1: Int, s2: Int, s3: Int, crossinline initialize: (Int, Int, Int, Int) -> Float): Tensor4 =
    NTensor(s0) { g -> Tensor3(s1, s2, s3) { h, i, j -> initialize(g, h, i, j) } }

// TODO: doc
inline fun Tensor4(s0: Int, s1: Int, s2: Int, s3: Int, crossinline initialize: (Int, Int, Int, Int) -> Float): Tensor4 =
    NTensor(s0, s1, s2, s3)

// TODO: doc
inline fun Tensor4(size: Int, crossinline initialize: (i: Int) -> Tensor3): Tensor4 =
    NTensor(size, initialize)

// TODO: doc
fun NTensor(s0: Int, s1: Int, s2: Int, s3: Int): Tensor4 =
    NTensor(s0) { NTensor(s1, s2, s3) }

// TODO: doc
fun Tensor4(s0: Int, s1: Int, s2: Int, s3: Int): Tensor4 =
    NTensor(s0) { Tensor3(s1, s2, s3) }

// TODO: doc
fun randomTensor(s0: Int, s1: Int, s2: Int, s3: Int): Tensor4 =
    NTensor(s0) { NTensor(s1) { randomMatrix(s2, s3) } }
