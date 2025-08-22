package kannoo.math

data class Dimensions(val height: Int, val width: Int) {
    init {
        if (height < 0 || width < 0)
            throw IllegalArgumentException("Dimensions cannot be negative")
    }

    fun toShape(): Shape =
        Shape(height, width)
}
