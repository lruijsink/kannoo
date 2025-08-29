package kannoo.vulkan

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
