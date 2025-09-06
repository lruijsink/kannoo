package kannoo.vulkan

import kannoo.core.ActivationFunction
import kannoo.impl.Linear
import kannoo.impl.Logistic
import kannoo.impl.ReLU
import org.lwjgl.PointerBuffer
import org.lwjgl.system.MemoryStack
import org.lwjgl.system.MemoryUtil
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.Paths
import java.nio.file.StandardOpenOption

fun readFileToNative(fileName: String): ByteBuffer =
    FileChannel.open(Paths.get(fileName), StandardOpenOption.READ).use { fc ->
        val fileSize = fc.size()
        val buffer = MemoryUtil.memAlloc(fileSize.toInt())
        while (buffer.hasRemaining()) fc.read(buffer)
        buffer.flip()
        return buffer
    }

fun MemoryStack.pointers(strings: Iterable<String>): PointerBuffer =
    pointers(*strings.map { UTF8(it) }.toTypedArray())

fun Int.divCeil(divisor: Int): Int =
    (this + divisor - 1) / divisor

val ActivationFunction.vulkanShaderId
    get(): Int = when (this) {
        is Linear -> 0
        is ReLU -> 1
        is Logistic -> 2
        else -> throw IllegalStateException("Activation function $this not yet supported")
    }
