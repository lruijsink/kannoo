
import kannoo.example.MNIST
import kannoo.math2.Tensor
import kannoo.math2.exp
import kannoo.math2.square
import kannoo.math2.vector

fun main() {
    MNIST()

    val v = vector(1, 2, 3).copy()
    val t = v as Tensor

    square(exp(vector(1, 2))) + vector(3, 4)
}
