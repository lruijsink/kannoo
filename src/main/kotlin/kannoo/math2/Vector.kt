package kannoo.math2

/**
 * 1-dimensional [Tensor] of elements (scalars). This is the lowest supported rank of [Tensor] and so [slices] is
 * inaccessible.
 */
@JvmInline
value class Vector(val elements: FloatArray) : Tensor {
    /**
     * Vectors are tensors of rank 1.
     */
    override val rank get() = 1

    /**
     * A vector tensor cannot be further subdivided into rank 0 slices (its individual elements). This field should not
     * be accessed.
     *
     * @throws VectorSlicesAccessException
     */
    override val slices get() = throw VectorSlicesAccessException()

    override val size: Int get() = elements.size

    operator fun get(index: Int): Float =
        elements[index]

    operator fun set(index: Int, value: Float) {
        elements[index] = value
    }

    override operator fun plus(t: Tensor): Vector =
        if (t !is Vector || t.size != this.size) throw IllegalArgumentException("Requires equal size vectors")
        else Vector(size) { i -> this[i] + t[i] }

    override operator fun minus(t: Tensor): Vector =
        if (t !is Vector || t.size != this.size) throw IllegalArgumentException("Requires equal size vectors")
        else Vector(size) { i -> this[i] - t[i] }

    override operator fun times(s: Float): Vector =
        map { it * s }

    override operator fun div(s: Float): Vector =
        map { it / s }

    override operator fun plusAssign(t: Tensor) {
        if (t !is Vector || t.size != this.size) throw IllegalArgumentException("Requires equal size vectors")
        for (i in 0 until size) this[i] += t[i]
    }

    override operator fun minusAssign(t: Tensor) {
        if (t !is Vector || t.size != this.size) throw IllegalArgumentException("Requires equal size vectors")
        for (i in 0 until size) this[i] -= t[i]
    }

    override operator fun timesAssign(s: Float) {
        mapAssign { it * s }
    }

    override operator fun divAssign(s: Float) {
        mapAssign { it / s }
    }

    override fun copy(): Vector =
        Vector(elements.copyOf())

    override fun map(transform: (Float) -> Float): Vector =
        Vector(size) { i -> transform(this[i]) }

    override fun mapAssign(transform: (Float) -> Float) {
        for (i in 0 until size) this[i] = transform(this[i])
    }
}

inline fun Vector(size: Int, crossinline init: (index: Int) -> Float): Vector =
    Vector(FloatArray(size) { i -> init(i) })

fun vector(vararg elements: Float): Vector =
    Vector(elements)

fun vector(vararg elements: Number): Vector =
    Vector(elements.size) { elements[it].toFloat() }

/**
 * Thrown when attempting to access [Vector.slices]
 */
class VectorSlicesAccessException :
    IllegalAccessException("Attempted to access slices of a vector, this operation is not supported")
