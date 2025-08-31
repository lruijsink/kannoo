package kannoo.math

/**
 * Tensor composed of [slices] with [rank] N - 1. There are two types of composite tensor:
 * - [Matrix] (rank 2)
 * - [NTensor] (rank 3+)
 *
 * The only non-[Composite] tensor type is [Vector].
 */
interface Composite : Tensor {

    /**
     * Slices that make up this tensor, themselves tensors of rank [rank]` - 1`
     */
    val slices: Array<out Tensor>

    /**
     * @param index Slice index to get
     *
     * @return Slice at index [index]
     */
    operator fun get(index: Int): Tensor

    /**
     * @param index Slice index to set
     *
     * @param slice New slice value
     */
    operator fun set(index: Int, slice: Tensor)
}

// TODO: doc
fun Composite(slices: List<Tensor>): Composite {
    if (slices.isEmpty()) throw IllegalArgumentException("Composite cannot be empty")
    if (slices.any { it.shape != slices[0].shape }) throw IllegalArgumentException("All slices must have same shape")

    @Suppress("UNCHECKED_CAST")
    return when (slices[0].rank) {
        1 -> Matrix((slices as List<Vector>).toTypedArray())
        2 -> Tensor3((slices as List<Matrix>).toTypedArray())
        3 -> Tensor4((slices as List<Tensor3>).toTypedArray())
        else -> TODO("Not yet implemented")
    }
}


// TODO: doc
inline fun Composite(size: Int, crossinline init: (Int) -> Tensor): Composite =
    if (size <= 0) throw IllegalArgumentException("Composite cannot be empty")
    else Composite(List(size) { init(it) })
