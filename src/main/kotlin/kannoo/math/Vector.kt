package kannoo.math

@JvmInline
value class Vector(val elements: FloatArray) {
    constructor(size: Int) : this(FloatArray(size))
    constructor(size: Int, init: (i: Int) -> Float) : this(FloatArray(size, init))

    val size
        get(): Int = elements.size

    val scalars
        get(): FloatArray = elements

    operator fun get(index: Int): Float =
        elements[index]

    operator fun set(index: Int, value: Float) {
        elements[index] = value
    }

    operator fun plus(rhs: Vector): Vector =
        zipMap(rhs) { a, b -> a + b }

    operator fun minus(rhs: Vector): Vector =
        zipMap(rhs) { a, b -> a - b }

    operator fun times(scalar: Float): Vector =
        transform { it * scalar }

    operator fun div(scalar: Float): Vector =
        transform { it / scalar }

    operator fun plusAssign(rhs: Vector) {
        if (size != rhs.size) throw IllegalArgumentException("Vectors must have same size")
        for (i in 0 until size) this[i] += rhs[i]
    }

    operator fun minusAssign(rhs: Vector) {
        if (size != rhs.size) throw IllegalArgumentException("Vectors must have same size")
        for (i in 0 until size) this[i] -= rhs[i]
    }

    operator fun timesAssign(rhs: Float) {
        for (i in 0 until size) this[i] *= rhs
    }

    fun sum(): Float =
        elements.sum()

    fun min(): Float =
        elements.min()

    fun max(): Float =
        elements.max()

    fun zipMap(rhs: Vector, fn: (Float, Float) -> Float): Vector =
        if (size != rhs.size) throw IllegalArgumentException("Vectors must have same size")
        else Vector(size) { fn(this[it], rhs[it]) }

    inline fun zipSumOf(rhs: Vector, fn: (Float, Float) -> Float): Float {
        if (size != rhs.size) throw IllegalArgumentException("Vectors must have same size")
        var sum = 0f
        for (i in 0 until size) sum += fn(this[i], rhs[i])
        return sum
    }

    fun transform(fn: (Float) -> Float): Vector =
        Vector(size) { fn(this[it]) }

    fun square(): Vector =
        transform { it * it }

    fun copyInto(destination: Vector) {
        if (size != destination.size) throw IllegalArgumentException("Vectors must have same size")
        elements.copyInto(destination.elements)
    }

    override fun toString(): String = elements.toList().toString()
}

fun emptyVector(): Vector =
    vectorOf()

fun vectorOf(vararg vs: Float) =
    Vector(floatArrayOf(*vs))

fun randomVector(size: Int): Vector =
    Vector(size) { randomFloat() }

fun hadamard(a: Vector, b: Vector) =
    a.zipMap(b) { x, y -> x * y }

inline fun <T> Iterable<T>.sumOfVector(selector: (T) -> Vector): Vector {
    val first = this.firstOrNull() ?: return emptyVector()
    val sum = selector(first)
    this.forEachIndexed { i, el ->
        if (i > 0) sum += selector(el)
    }
    return sum
}

fun Iterable<Vector>.sum(): Vector {
    val sum = this.firstOrNull() ?: return emptyVector()
    this.forEachIndexed { i, matrix ->
        if (i > 0) sum += matrix
    }
    return sum
}

operator fun Float.times(vector: Vector): Vector = vector * this
