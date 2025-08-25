@file:Suppress("FunctionName")

package kannoo.math

/**
 * Rank 4 tensor; an [NTensor] with [Tensor3] slices.
 */
typealias Tensor4 = NTensor<Tensor3>

/**
 * @param s2 Dimension 1 size

 * @param s3 Dimension 2 size

 * @param s4 Dimension 3 size

 * @param s4 Dimension 4 size

 * @param initialize Elementwise function

 * @return [Tensor4] of shape (s1, s2, s3, s4) initialized with:
 *
 * `T[i, j, k, l]` = `initialize(i, j, k, l)`
 */
inline fun NTensor(s1: Int, s2: Int, s3: Int, s4: Int, crossinline initialize: (Int, Int, Int, Int) -> Float): Tensor4 =
    NTensor(s1) { i -> Tensor3(s2, s3, s4) { j, k, l -> initialize(i, j, k, l) } }

/**
 * @param s2 Dimension 1 size

 * @param s3 Dimension 2 size

 * @param s4 Dimension 3 size

 * @param s4 Dimension 4 size

 * @param initialize Elementwise function

 * @return [Tensor4] of shape (s1, s2, s3, s4) initialized with:
 *
 * `T[i, j, k, l]` = `initialize(i, j, k, l)`
 */
inline fun Tensor4(s1: Int, s2: Int, s3: Int, s4: Int, crossinline initialize: (Int, Int, Int, Int) -> Float): Tensor4 =
    NTensor(s1, s2, s3, s4, initialize)

/**
 * @param size Dimension 1 size, the number of [Tensor3] slices to instantiate

 * @param initialize Slice initialization function

 * @return [Tensor4] of [size] [Tensor3] slices such that:
 *
 * `T[i]` = `initialize(i)`
 *
 * @throws IncompatibleShapeException If [initialize] produces slices of differing shapes
 */
inline fun Tensor4(size: Int, crossinline initialize: (i: Int) -> Tensor3): Tensor4 =
    NTensor(size, initialize)

/**
 * @param s1 Dimension 1 size

 * @param s2 Dimension 2 size

 * @param s3 Dimension 3 size

 * @param s4 Dimension 4 size

 * @return [Tensor4] of shape (s1, s2, s3, s4), initialized to zeroes
 */
fun NTensor(s1: Int, s2: Int, s3: Int, s4: Int): Tensor4 =
    NTensor(s1) { NTensor(s2, s3, s4) }

/**
 * @param s1 Dimension 1 size

 * @param s2 Dimension 2 size

 * @param s3 Dimension 3 size

 * @param s4 Dimension 4 size

 * @return [Tensor4] of shape (s1, s2, s3, s4), initialized to zeroes
 */
fun Tensor4(s1: Int, s2: Int, s3: Int, s4: Int): Tensor4 =
    NTensor(s1) { Tensor3(s2, s3, s4) }

/**
 * @param s1 Dimension 1 size

 * @param s2 Dimension 2 size

 * @param s3 Dimension 3 size

 * @param s4 Dimension 4 size

 * @return [Tensor4] of shape (s1, s2, s3, s4), with elements randomized by [randomSignedFloat]
 */
fun randomTensor(s1: Int, s2: Int, s3: Int, s4: Int): Tensor4 =
    NTensor(s1) { NTensor(s2) { randomMatrix(s3, s4) } }
