package kannoo

typealias Vector = DoubleArray

fun emptyVector(): Vector =
    doubleArrayOf()

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
