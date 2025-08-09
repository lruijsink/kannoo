package kannoo.io

import kannoo.core.InnerLayer
import kannoo.impl.DenseLayer
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.OutputStream
import javax.imageio.ImageIO
import kotlin.math.roundToInt

private val positive = Color(0xff, 0xaa, 0x33)
private val negative = Color(0x33, 0xaa, 0xff)

fun OutputStream.writeLayerAsRGB(layer: InnerLayer) {
    val w = (layer as DenseLayer).weights
    val wMin = w.rowVectors.minOf { it.min() }
    val wMax = w.rowVectors.maxOf { it.max() }

    operator fun Color.times(s: Float): Color =
        Color((red * s).roundToInt(), (green * s).roundToInt(), (blue * s).roundToInt())

    fun Float.colorize(): Color =
        if (this < 0) negative * (this / wMin) else positive * (this / wMax)

    val b = BufferedImage(w.cols, w.rows, BufferedImage.TYPE_INT_RGB)
    w.forEachIndexed { row, col -> b.setRGB(col, row, w[row][col].colorize().rgb) }
    ImageIO.write(b, "png", this)
}
