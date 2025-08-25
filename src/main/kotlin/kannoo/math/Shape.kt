package kannoo.math

/**
 * Describes a tensor shape.
 *
 * See also [Tensor.shape]
 *
 * @param dimensions Dimensions from highest rank to lowest
 */
class Shape(val dimensions: List<Int>) {

    constructor(vararg dimensions: Int) :
            this(dimensions.toList())

    /**
     * Construct the shape of a [Composite] tensor.
     *
     * @param compositeSize Composite tensor (highest rank) size
     *
     * @param sliceShape Shape of the composite tensor's slices
     */
    constructor(compositeSize: Int, sliceShape: Shape) :
            this(listOf(compositeSize) + sliceShape.dimensions)

    /**
     * Total number of elements in the tensor, across all ranks. Equivalent to multiplying the size of each slice. For
     * example: a tensor of 3 matrix slices with dimensions 2 x 5 has 3 x 2 x 5 = 30 total elements.
     */
    val totalElements: Int get() =
        dimensions.reduce { x, y -> x * y }

    /**
     * Rank = number of dimensions of this shape
     */
    val rank: Int get() =
        dimensions.size

    /**
     * Shape of this shape's slices. Has rank N - 1 and drops the first dimension. For example:
     *
     * `Shape(3, 8, 6).sliceShape = Shape(8, 6)`
     */
    val sliceShape: Shape get() =
        Shape(dimensions.drop(1))

    /**
     * Shorthand for `dimensions[index]`
     *
     * @return Dimension [index]'s size
     */
    operator fun get(index: Int): Int =
        dimensions[index]

    /**
     * @return New tensor, with elements initialized to zero, with dimensions equal to this shape.
     */
    fun createTensor(): Tensor = when (rank) {
        1 -> Vector(this[0])
        2 -> Matrix(this[0], this[1])
        3 -> NTensor(this[0]) { Matrix(this[1], this[2]) }
        else -> NTensor(this[0]) { sliceShape.createTensor() as NTensor<*> }
    }

    operator fun component1(): Int =
        dimensions[0]

    operator fun component2(): Int =
        dimensions[1]

    operator fun component3(): Int =
        dimensions[2]

    operator fun component4(): Int =
        dimensions[3]

    operator fun component5(): Int =
        dimensions[4]

    override fun equals(other: Any?): Boolean =
        other is Shape && dimensions == other.dimensions

    override fun hashCode(): Int =
        dimensions.hashCode()

    override fun toString(): String =
        dimensions.toString()
}
