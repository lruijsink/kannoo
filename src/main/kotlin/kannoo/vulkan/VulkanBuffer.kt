package kannoo.vulkan

import org.lwjgl.system.MemoryStack.stackPush
import org.lwjgl.system.MemoryUtil.memByteBuffer
import org.lwjgl.vulkan.VK10.vkDestroyBuffer
import org.lwjgl.vulkan.VK10.vkFreeMemory
import org.lwjgl.vulkan.VK10.vkMapMemory
import org.lwjgl.vulkan.VK10.vkUnmapMemory
import java.nio.ByteBuffer

class VulkanBuffer(
    val size: Long,
    val vulkan: Vulkan,
    val handle: Long,
    val memory: Long,
) {
    val mapped: ByteBuffer = map()

    private fun map(): ByteBuffer = stackPush().use { stack ->
        val pMappedMemory = stack.mallocPointer(1)
        vkMapMemory(vulkan.device, memory, 0, size, 0, pMappedMemory).orThrow()
        memByteBuffer(pMappedMemory.get(0), size.toInt())
    }

    fun destroy() {
        vkUnmapMemory(vulkan.device, memory)
        vkFreeMemory(vulkan.device, memory, null)
        vkDestroyBuffer(vulkan.device, handle, null)
    }
}
