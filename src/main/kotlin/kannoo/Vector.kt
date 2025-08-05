package kannoo

@JvmInline
value class Vector(private val vs: DoubleArray) {
    constructor(size: Int) : this(DoubleArray(size))
    constructor(size: Int, init: (i: Int) -> Double) : this(DoubleArray(size, init))

    val size
        get(): Int = vs.size

    val scalars
        get(): DoubleArray = vs

    operator fun get(index: Int): Double =
        vs[index]

    operator fun set(index: Int, value: Double) {
        vs[index] = value
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
        vs.sum()

    fun zipMap(rhs: Vector, fn: (Double, Double) -> Double): Vector =
        if (size != rhs.size) throw IllegalArgumentException("Vectors must have same size")
        else Vector(size) { fn(this[it], rhs[it]) }

    fun transform(fn: (Double) -> Double): Vector =
        Vector(size) { fn(this[it]) }

    fun square(): Vector =
        transform { it * it }

    fun copyInto(destination: Vector) {
        if (size != destination.size) throw IllegalArgumentException("Vectors must have same size")
        vs.copyInto(destination.vs)
    }
}

fun emptyVector(): Vector =
    vectorOf()

fun vectorOf(vararg vs: Double) =
    Vector(doubleArrayOf(*vs))

fun randomVector(size: Int): Vector =
    Vector(size) { randomDouble() }

fun hadamard(a: Vector, b: Vector) =
    if (a.size != b.size) throw IllegalArgumentException("Vectors must have same size")
    else Vector(a.size) { a[it] * b[it] }

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
