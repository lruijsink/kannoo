import kannoo.math.Matrix
import kannoo.vulkan.Vulkan
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.FileOutputStream
import javax.imageio.ImageIO
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt
import kotlin.system.measureTimeMillis

fun main() {
    println("Creating Vulkan instance")
    val vulkan: Vulkan?
    val msInit = measureTimeMillis { vulkan = Vulkan(shaderFile = "shaders/comp.spv") }
    println("Vulkan instance successfully created (took $msInit ms)")
    vulkan!!

    println("Setting input")
    val xO = vulkan.inputWidth / 2
    val yO = vulkan.inputHeight / 2
    vulkan.setInput(
        Matrix(vulkan.inputHeight, vulkan.inputWidth) { y, x ->
            val rx = (xO - x).toFloat() / xO
            val ry = (yO - y).toFloat() / yO
            if (sqrt(rx*rx + ry*ry) < 1f) 1f else 0f
        }.flatten().elements,
    )
    println("Input successfully set")

    println("Running command buffer")
    val msRun = measureTimeMillis { vulkan.runCommandBuffer() }
    println("Command buffer successfully run (took $msRun ms)")

    println("Retrieving rendered image")
    var scalars = FloatArray(1)
    val msGet = measureTimeMillis { scalars = vulkan.getRenderedImage() }
    println("Rendered image successfully retrieved (took $msGet ms)")

    println("Serializing image")
    val bufferedImage = BufferedImage(vulkan.outputWidth, vulkan.outputHeight, BufferedImage.TYPE_INT_RGB)
    val msRender = measureTimeMillis {
        var i = 0
        for (y in 0 until vulkan.outputHeight) {
            for (x in 0 until vulkan.outputWidth) {
                val c = Color(
                    min(255, (scalars[i++] * 255f).roundToInt()),
                    min(255, (scalars[i++] * 255f).roundToInt()),
                    min(255, (scalars[i++] * 255f).roundToInt()),
                    min(255, (scalars[i++] * 255f).roundToInt()),
                )
                bufferedImage.setRGB(x, y, c.rgb)
            }
        }
    }
    ImageIO.write(bufferedImage, "png", FileOutputStream("mandelbrot.png"))
    println("Image successfully serialized (took $msRender ms)")

    println("Destroying Vulkan instance")
    vulkan.destroy()
    println("Vulkan instance successfully destroyed")
}
