package kannoo.math

/**
 * Describes a tensor shape.
 *
 * See also [Tensor.shape]
 *
 * @param dimensions Dimensions from highest rank to lowest
 */
data class Shape(val dimensions: List<Int>) {

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
     * @return New tensor, with elements initialized to zero, with dimensions equal to this shape.
     */
    fun createTensor(): TensorBase = when (dimensions.size) {
        1 -> Vector(dimensions[0])
        2 -> Matrix(dimensions[0], dimensions[1])
        3 -> NTensor(dimensions[0]) { Matrix(dimensions[1], dimensions[2]) }
        else -> NTensor(dimensions[0]) { Shape(dimensions.drop(1)).createTensor() as NTensor<*> }
    }
}
