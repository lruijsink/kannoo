
import kannoo.math.randomVector
import kannoo.math2.Vector
import kotlin.random.Random
import kotlin.system.measureTimeMillis

fun time(label: String, fn: () -> Unit) {
    println(label + ": " + measureTimeMillis { fn() } + " ms")
}

fun main() {
    val oldVectors = Array(50_000) { randomVector(10_000) }
    time("Old vector add") {
        val res = oldVectors[0]
        for (i in 1 until oldVectors.size) res += oldVectors[i]
    }

    val newVectors = Array(50_000) { Vector(10_000) { Random.nextFloat() } }
    time("New vector add") {
        val res = newVectors[0]
        for (i in 1 until newVectors.size) res += newVectors[i]
    }
}
