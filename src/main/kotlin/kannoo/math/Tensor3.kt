@file:Suppress("FunctionName")

package kannoo.math

/**
 * Rank 3 tensor; an [NTensor] with [Matrix] slices.
 */
typealias Tensor3 = NTensor<Matrix>

/**
 * @param s1 Dimension 1 size

 * @param s2 Dimension 2 size

 * @param s3 Dimension 3 size

 * @param initialize Elementwise function

 * @return [Tensor3] of shape (s1, s2, s3) initialized with:
 *
 * `T[i, j, k]` = `initialize(i, j, k)`
 */
inline fun NTensor(s1: Int, s2: Int, s3: Int, crossinline initialize: (Int, Int, Int) -> Float): Tensor3 =
    NTensor(s1) { h -> Matrix(s2, s3) { i, j -> initialize(h, i, j) } }

/**
 * @param s1 Dimension 1 size

 * @param s2 Dimension 2 size

 * @param s3 Dimension 3 size

 * @param initialize Elementwise function

 * @return [Tensor3] of shape (s1, s2, s3) initialized with:
 *
 * `T[i, j, k]` = `initialize(i, j, k)`
 */
inline fun Tensor3(s1: Int, s2: Int, s3: Int, crossinline initialize: (Int, Int, Int) -> Float): Tensor3 =
    NTensor(s1, s2, s3, initialize)

/**
 * @param size Dimension 1 size, the number of [Matrix] slices to instantiate

 * @param initialize Slice initialization function

 * @return [Tensor3] of [size] [Matrix] slices such that:
 *
 * `T[i]` = `initialize(i)`
 *
 * @throws IncompatibleShapeException If [initialize] produces slices of differing shapes
 */
inline fun Tensor3(size: Int, crossinline initialize: (i: Int) -> Matrix): Tensor3 =
    NTensor(size, initialize)

/**
 * @param s1 Dimension 1 size

 * @param s2 Dimension 2 size

 * @param s3 Dimension 3 size

 * @return [Tensor3] of shape (s1, s2, s3), initialized to zeroes
 */
fun NTensor(s1: Int, s2: Int, s3: Int): Tensor3 =
    NTensor(s1) { Matrix(s2, s3) }

/**
 * @param s1 Dimension 1 size

 * @param s2 Dimension 2 size

 * @param s3 Dimension 3 size

 * @return [Tensor3] of shape (s1, s2, s3), initialized to zeroes
 */
fun Tensor3(s1: Int, s2: Int, s3: Int): Tensor3 =
    Tensor3(s1) { Matrix(s2, s3) }

/**
 * @param s1 Dimension 1 size

 * @param s2 Dimension 2 size

 * @param s3 Dimension 3 size

 * @return [Tensor3] of shape (s1, s2, s3), with elements randomized by [randomSignedFloat]
 */
fun randomTensor(s1: Int, s2: Int, s3: Int): Tensor3 =
    NTensor(s1) { randomMatrix(s2, s3) }

// TODO: Generalize broadcasting
/**
 * @param vector Vector to broadcast
 *
 * @return New tensor T with [vector] added to each row vector (dimension 3 tensor) in this tensor
 *
 * @throws IncompatibleShapeException If this tensor's last dimension's size differs from [vector]'s
 */
infix fun Tensor3.broadcastPlus(vector: Vector): Tensor3 =
    if (this.size != vector.size)
        throw IncompatibleShapeException("Tensor and vector sizes must match")
    else
        Tensor3(this.size) { i -> this[i].map { it + vector[i] } }
