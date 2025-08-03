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

    operator fun minus(rhs: Vector): Vector =
        if (size != rhs.size) throw IllegalArgumentException("Vectors must have same size")
        else zipMap(rhs) { a, b -> a - b }

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
