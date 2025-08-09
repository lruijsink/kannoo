package kannoo.io

import kannoo.core.InnerLayer
import kannoo.impl.DenseLayer
import kannoo.math.Matrix
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.OutputStream
import javax.imageio.ImageIO
import kotlin.math.ceil
import kotlin.math.roundToInt
import kotlin.math.sqrt

private val positive = Color(0xff, 0xaa, 0x33)
private val negative = Color(0x33, 0xaa, 0xff)

private operator fun Color.times(s: Float): Color =
    Color((red * s).roundToInt(), (green * s).roundToInt(), (blue * s).roundToInt())

fun OutputStream.writeLayerAsRGB(layer: InnerLayer) =
    drawMatrix((layer as DenseLayer).weights)

fun OutputStream.drawMatrix(m: Matrix) {
    val b = BufferedImage(m.cols, m.rows, BufferedImage.TYPE_INT_RGB)
    b.drawMatrix(m, 0, 0)
    ImageIO.write(b, "png", this)
}

fun BufferedImage.drawMatrix(m: Matrix, offsetX: Int = 0, offsetY: Int = 0) {
    val wMin = m.rowVectors.minOf { it.min() }
    val wMax = m.rowVectors.maxOf { it.max() }

    fun Float.colorize(): Color =
        if (this < 0) negative * (this / wMin) else positive * (this / wMax)

    m.forEachIndexed { row, col -> setRGB(col + offsetX, row + offsetY, m[row][col].colorize().rgb) }
}

fun OutputStream.writeMatricesAsRGB(matrices: List<Matrix>, padding: Int = 0) {
    val span = ceil(sqrt(matrices.size.toFloat())).toInt()
    val mW = matrices[0].cols
    val mH = matrices[0].rows
    val b = BufferedImage(
        mW * span + (span - 1) * padding,
        mH * span + (span - 1) * padding,
        BufferedImage.TYPE_INT_RGB,
    )
    matrices.forEachIndexed { i, m ->
        b.drawMatrix(m, offsetX = (i % span) * (mW + padding), offsetY = (i / span) * (mH + padding))
    }
    ImageIO.write(b, "png", this)
}
