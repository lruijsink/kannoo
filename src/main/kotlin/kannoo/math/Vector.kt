package kannoo.math

@JvmInline
value class Vector(val elements: DoubleArray) {
    constructor(size: Int) : this(DoubleArray(size))
    constructor(size: Int, init: (i: Int) -> Double) : this(DoubleArray(size, init))

    val size
        get(): Int = elements.size

    val scalars
        get(): DoubleArray = elements

    operator fun get(index: Int): Double =
        elements[index]

    operator fun set(index: Int, value: Double) {
        elements[index] = value
    }

    operator fun plus(rhs: Vector): Vector =
        zipMap(rhs) { a, b -> a + b }

    operator fun minus(rhs: Vector): Vector =
        zipMap(rhs) { a, b -> a - b }

    operator fun times(scalar: Double): Vector =
        transform { it * scalar }

    operator fun div(scalar: Double): Vector =
        transform { it / scalar }

    operator fun plusAssign(rhs: Vector) {
        if (size != rhs.size) throw IllegalArgumentException("Vectors must have same size")
        for (i in 0 until size) this[i] += rhs[i]
    }

    operator fun minusAssign(rhs: Vector) {
        if (size != rhs.size) throw IllegalArgumentException("Vectors must have same size")
        for (i in 0 until size) this[i] -= rhs[i]
    }

    operator fun timesAssign(rhs: Double) {
        for (i in 0 until size) this[i] *= rhs
    }

    fun sum(): Double =
        elements.sum()

    fun min(): Double =
        elements.min()

    fun max(): Double =
        elements.max()

    fun zipMap(rhs: Vector, fn: (Double, Double) -> Double): Vector =
        if (size != rhs.size) throw IllegalArgumentException("Vectors must have same size")
        else Vector(size) { fn(this[it], rhs[it]) }

    inline fun zipSumOf(rhs: Vector, fn: (Double, Double) -> Double): Double {
        if (size != rhs.size) throw IllegalArgumentException("Vectors must have same size")
        var sum = 0.0
        for (i in 0 until size) sum += fn(this[i], rhs[i])
        return sum
    }

    fun transform(fn: (Double) -> Double): Vector =
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

fun vectorOf(vararg vs: Double) =
    Vector(doubleArrayOf(*vs))

fun randomVector(size: Int): Vector =
    Vector(size) { randomDouble() }

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

operator fun Double.times(vector: Vector): Vector = vector * this
