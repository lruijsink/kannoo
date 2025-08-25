package kannoo.math

class Padding(
    val height: Int,
    val width: Int,
    val scheme: PaddingScheme,
)

interface PaddingScheme {
    fun pad(i: Int, j: Int, input: Matrix): Float
    fun map(i: Int, s: Int): Int
}

object ZeroPadding : PaddingScheme {
    override fun pad(i: Int, j: Int, input: Matrix): Float =
        if (0 <= i && i < input.rows && 0 <= j && j < input.cols) input[i, j]
        else 0f

    override fun map(i: Int, s: Int): Int =
        if (i < 0 || i >= s) -1
        else i
}

object CircularPadding : PaddingScheme {
    override fun pad(i: Int, j: Int, input: Matrix): Float =
        input[wrap(i, input.rows), wrap(j, input.rows)]

    override fun map(i: Int, s: Int): Int =
        wrap(i, s)

    private fun wrap(n: Int, m: Int): Int = when {
        n < 0 -> m + n
        n >= m -> n - m
        else -> n
    }
}

object ReflectionPadding : PaddingScheme {
    override fun pad(i: Int, j: Int, input: Matrix): Float =
        input[reflect(i, input.rows), reflect(j, input.cols)]

    override fun map(i: Int, s: Int): Int =
        reflect(i, s)

    private fun reflect(n: Int, m: Int): Int = when {
        n < 0 -> -n
        n >= m -> 2 * m - n - 2
        else -> n
    }
}

object ReplicationPadding : PaddingScheme {
    override fun pad(i: Int, j: Int, input: Matrix): Float =
        input[i.coerceIn(0, input.rows - 1), j.coerceIn(0, input.cols - 1)]

    override fun map(i: Int, s: Int): Int =
        i.coerceIn(0, s - 1)
}
