package kannoo

@JvmInline
value class Vector(private val vs: DoubleArray) {
    constructor(size: Int) : this(DoubleArray(size))
    constructor(size: Int, init: (i: Int) -> Double) : this(DoubleArray(size, init))

    val size
        get(): Int = vs.size

    operator fun get(index: Int): Double =
        vs[index]

    operator fun set(index: Int, value: Double) {
        vs[index] = value
    }

    fun sum(): Double =
        vs.sum()

    fun <T> map(fn: (Double) -> T) =
        vs.map(fn)

    fun copyInto(destination: Vector) {
        vs.copyInto(destination.vs)
    }
}

fun emptyVector(): Vector =
    Vector(doubleArrayOf())

fun vectorOf(vararg vs: Double) =
    Vector(doubleArrayOf(*vs))

fun randomVector(size: Int): Vector =
    Vector(size) { randomDouble() }

fun Vector.mapDouble(fn: (Double) -> Double): Vector =
    Vector(size) { fn(this[it]) }

fun square(v: Vector): Vector =
    v.mapDouble { it * it }

infix fun Vector.sub(v: Vector): Vector =
    zipMap(this, v) { a, b -> a - b }

fun Vector.addInPlace(v: Vector) {
    if (size != v.size) throw IllegalArgumentException("Must be equal size")
    for (i in 0 until size) this[i] += v[i]
}

fun Vector.subInPlace(v: Vector) {
    if (size != v.size) throw IllegalArgumentException("Must be equal size")
    for (i in 0 until size) this[i] -= v[i]
}

fun Vector.mulInPlace(s: Double) {
    for (i in 0 until size) this[i] *= s
}

fun zipMap(a: Vector, b: Vector, f: (Double, Double) -> Double) =
    if (a.size != b.size) throw IllegalArgumentException("Must be equal size")
    else Vector(a.size) { f(a[it], b[it]) }

fun hadamard(a: Vector, b: Vector) =
    if (a.size != b.size) throw IllegalArgumentException("Must be equal size")
    else Vector(a.size) { a[it] * b[it] }
